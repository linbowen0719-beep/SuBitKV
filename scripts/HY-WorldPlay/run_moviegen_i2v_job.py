#!/usr/bin/env python3
"""Run one frozen Movie Gen I2V input through one HY-WorldPlay KV method."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.rvq import rvq_codebook_directory_metadata  # noqa: E402

DEFAULT_MODEL = Path(os.environ["QVG_HY8B_MODEL"]) if os.environ.get("QVG_HY8B_MODEL") else None
DEFAULT_ACTION = Path(os.environ["QVG_HY_ACTION_CKPT"]) if os.environ.get("QVG_HY_ACTION_CKPT") else None
DEFAULT_PYTHON = Path(os.environ.get("QVG_PYTHON", sys.executable))
METHOD_CODEC = {
    "collect": "none",
    "collect_residual": "none",
    "calibrate_avg2": "none",
    "bf16": "none",
    "ours_rvq_avg1": "rvq_ours_avg1",
    "ours_rvq_avg2": "rvq_ours_avg2",
    "ours_rvq_avg3": "rvq_ours_avg3",
    "ours_rvq_avg4": "rvq_ours_avg4",
    "ours_rvq_avg0": "rvq_ours_avg0",
    "ours_rvq_avg0p2_alt": "rvq_ours_avg0p2_alt",
    "ours_rvq_avg0p5": "rvq_ours_avg0p5",
    "kivi_int2": "kivi_int2",
    "kivi_int4": "kivi_int4",
    "turboquant_int2": "turboquant_int2",
    "turboquant_int4": "turboquant_int4",
    "quantvideo_int2": "quantvideo_int2",
    "quantvideo_int4": "quantvideo_int4",
}
OURS_METHODS = frozenset(name for name in METHOD_CODEC if name.startswith("ours_rvq_avg"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def query_gpu(index: int) -> dict[str, Any]:
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    name, uuid, total, used, util = [value.strip() for value in output.split(",", 4)]
    return {"index": index, "name": name, "uuid": uuid, "total_mib": int(total), "used_mib": int(used), "utilization": int(util)}


def compute_pids(index: int) -> set[int]:
    uuid = query_gpu(index)["uuid"]
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    result = set()
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[0] == uuid:
            result.add(int(fields[1]))
    return result


class NvmlMonitor:
    def __init__(self, gpu: int, initial_pids: set[int]) -> None:
        self.gpu = gpu
        self.initial_pids = set(initial_pids)
        self.peak_mib = 0
        self.samples = 0
        self.observed_pids: set[int] = set()
        self.external_pids: set[int] = set()
        self.allowed_root_pid: int | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(0.2):
            try:
                row = query_gpu(self.gpu)
                self.peak_mib = max(self.peak_mib, int(row["used_mib"]))
                self.samples += 1
                pids = compute_pids(self.gpu)
                self.observed_pids.update(pids)
                if self.allowed_root_pid is not None:
                    self.external_pids.update(
                        pid for pid in pids
                        if not _is_pid_or_descendant(pid, self.allowed_root_pid)
                    )
            except Exception:
                pass

    def start(self) -> None:
        self.thread.start()

    def allow_process_tree(self, pid: int) -> None:
        self.allowed_root_pid = int(pid)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)


def _parent_pid(pid: int) -> int | None:
    try:
        # /proc/PID/stat field 2 may contain spaces and parentheses, so split
        # only after the final closing parenthesis; field 4 is then index 1.
        tail = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return int(tail[1])
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        return None


def _is_pid_or_descendant(pid: int, root_pid: int) -> bool:
    current = int(pid)
    root_pid = int(root_pid)
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == root_pid:
            return True
        seen.add(current)
        parent = _parent_pid(current)
        if parent is None:
            return False
        current = parent
    return False


def record_for(manifest: dict, split: str, index: int) -> dict:
    matches = [row for row in manifest["records"] if row["split"] == split and int(row["split_index"]) == index]
    if len(matches) != 1:
        raise ValueError(f"manifest has {len(matches)} records for {split}[{index}]")
    return matches[0]


def summarize_events(events: list[dict]) -> dict[str, Any]:
    if not events:
        return {"count": 0, "total_seconds": 0.0, "by_history_spans": {}}
    values = [float(row["seconds"]) for row in events]
    by_span: dict[str, list[float]] = {}
    for row in events:
        by_span.setdefault(str(int(row.get("history_spans", 0))), []).append(float(row["seconds"]))
    return {
        "count": len(values),
        "total_seconds": sum(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "max_seconds": max(values),
        "by_history_spans": {
            key: {"count": len(group), "total_seconds": sum(group), "mean_seconds": statistics.mean(group), "max_seconds": max(group)}
            for key, group in sorted(by_span.items(), key=lambda item: int(item[0]))
        },
    }


def summarize_avg2_accounting(
    events: list[dict],
    codebook_storage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Collapse repeated denoise encodes into one 54-layer cache snapshot."""

    supported = {
        "hyworldplay_rvq_ours_avg2_accounting.v2",
        "hyworldplay_rvq_ours_avg2_accounting.v3",
        "hyworldplay_rvq_temporal_accounting.v4",
    }
    snapshots: dict[int, dict[int, dict[str, Any]]] = {}
    event_rows: dict[int, dict[int, dict[str, Any]]] = {}
    for event in events:
        accounting = event.get("post_truncation_accounting")
        if not isinstance(accounting, dict) or accounting.get("schema_version") not in supported:
            continue
        spans = int(event.get("history_spans", 0))
        layer = int(event["layer"])
        # Assignment intentionally keeps the last copy: official four-step
        # denoising rebuilds an equivalent history several times.
        snapshots.setdefault(spans, {})[layer] = accounting
        event_rows.setdefault(spans, {})[layer] = event
    if not snapshots:
        return None
    byte_fields = (
        "physical_local_k_bytes",
        "physical_local_v_bytes",
        "physical_residual_k_bytes",
        "physical_residual_v_bytes",
        "physical_rvq_base_k_bytes",
        "physical_rvq_base_v_bytes",
        "physical_rvq_indices_k_bytes",
        "physical_rvq_indices_v_bytes",
        "physical_rvq_scales_k_bytes",
        "physical_rvq_scales_v_bytes",
        "physical_rvq_early_stop_depth_k_bytes",
        "physical_rvq_early_stop_depth_v_bytes",
        "physical_rvq_other_tensor_k_bytes",
        "physical_rvq_other_tensor_v_bytes",
        "physical_shared_k_bytes",
        "physical_shared_v_bytes",
        "physical_payload_bytes",
    )
    codebook_bytes = int((codebook_storage or {}).get("combined_storage_bytes", 0))
    rows = []
    for spans in sorted(snapshots):
        by_layer = snapshots[spans]
        totals = {
            field: sum(int(row.get(field, 0)) for row in by_layer.values())
            for field in byte_fields
        }
        bf16_k_bytes = sum(
            int(accounting.get("bf16_k_bytes", event_rows[spans][layer].get("bf16_bytes", 0) // 2))
            for layer, accounting in by_layer.items()
        )
        bf16_v_bytes = sum(
            int(accounting.get("bf16_v_bytes", event_rows[spans][layer].get("bf16_bytes", 0) // 2))
            for layer, accounting in by_layer.items()
        )
        bf16_bytes = int(bf16_k_bytes + bf16_v_bytes)
        payload = int(totals["physical_payload_bytes"])
        residual_payload = int(
            totals["physical_residual_k_bytes"]
            + totals["physical_residual_v_bytes"]
            + totals["physical_shared_k_bytes"]
            + totals["physical_shared_v_bytes"]
        )
        scalar_values = int(bf16_bytes // 2)
        nominal_sum = sum(
            int(accounting.get("selected_k_bit_sum", 0))
            + int(accounting.get("selected_v_bit_sum", 0))
            for accounting in by_layer.values()
        )
        k_payload = int(totals["physical_local_k_bytes"] + totals["physical_shared_k_bytes"])
        v_payload = int(totals["physical_local_v_bytes"] + totals["physical_shared_v_bytes"])
        k_codebook_bytes = int((codebook_storage or {}).get("K", {}).get("codebook_storage_bytes", 0))
        v_codebook_bytes = int((codebook_storage or {}).get("V", {}).get("codebook_storage_bytes", 0))
        rows.append({
            "history_spans": spans,
            "layers": len(by_layer),
            "bf16_k_bytes": bf16_k_bytes,
            "bf16_v_bytes": bf16_v_bytes,
            "bf16_kv_bytes": bf16_bytes,
            **totals,
            "shared_codebook_bytes": codebook_bytes,
            "nominal_residual_avg_bits": (
                nominal_sum / (2.0 * len(by_layer) * spans)
                if by_layer and spans else None
            ),
            "physical_residual_bpp": (
                residual_payload * 8.0 / scalar_values
                if scalar_values else None
            ),
            "dynamic_kv_bpp": (
                payload * 8.0 / scalar_values if scalar_values else None
            ),
            "one_video_total_bpp": (
                (payload + codebook_bytes) * 8.0 / scalar_values
                if scalar_values else None
            ),
            "two_video_amortized_bpp": (
                (payload + codebook_bytes / 2.0) * 8.0 / scalar_values
                if scalar_values else None
            ),
            "compression_ratio_K_excluding_shared_codebook": (
                bf16_k_bytes / k_payload if k_payload else None
            ),
            "compression_ratio_V_excluding_shared_codebook": (
                bf16_v_bytes / v_payload if v_payload else None
            ),
            "compression_ratio_K_including_one_codebook_copy": (
                bf16_k_bytes / (k_payload + k_codebook_bytes)
                if k_payload + k_codebook_bytes else None
            ),
            "compression_ratio_V_including_one_codebook_copy": (
                bf16_v_bytes / (v_payload + v_codebook_bytes)
                if v_payload + v_codebook_bytes else None
            ),
            "compression_ratio_vs_bf16_excluding_shared_codebooks": (
                bf16_bytes / payload if payload else None
            ),
            "compression_ratio_vs_bf16_including_one_codebook_copy": (
                bf16_bytes / (payload + codebook_bytes)
                if payload + codebook_bytes else None
            ),
            "compression_ratio_vs_bf16_two_video_amortized": (
                bf16_bytes / (payload + codebook_bytes / 2.0)
                if payload + codebook_bytes else None
            ),
        })
    return {
        "schema_version": "hyworldplay_rvq_avg2_physical_summary.v1",
        "baseline": "BF16 K/V tensor payload",
        "repeated_denoise_rebuilds_deduplicated": True,
        "snapshots": rows,
        "terminal_snapshot": rows[-1],
        "codebook_storage": codebook_storage,
    }


def codebook_storage_metadata(root: Path | None) -> dict[str, Any] | None:
    if root is None:
        return None
    k_dir = root / "nonorm_k_l16c64_codebooks"
    v_dir = root / "nonorm_v_l16c64_codebooks"
    if not k_dir.is_dir() or not v_dir.is_dir():
        return None
    k = rvq_codebook_directory_metadata(k_dir, storage_dtype="int4")
    v = rvq_codebook_directory_metadata(v_dir, storage_dtype="int4")
    combined = int(k["codebook_storage_bytes"]) + int(v["codebook_storage_bytes"])
    return {
        "storage_basis": "canonical packed INT4 centroids plus per-centroid FP16 scale",
        "K": k,
        "V": v,
        "combined_storage_bytes": combined,
        "combined_storage_mib": combined / 1024**2,
    }


def probe(path: Path, ffprobe: str) -> dict[str, Any]:
    output = subprocess.check_output(
        [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)],
        text=True,
    )
    stream = json.loads(output)["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    return {"width": int(stream["width"]), "height": int(stream["height"]), "fps": int(numerator) / int(denominator), "frames": int(stream["nb_read_frames"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", choices=tuple(METHOD_CODEC), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--gpu", type=int, choices=tuple(range(8)), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-hash", default="96463403384b8194729d0de04424a7f2538c8694b030d54c259a44202c1d3464")
    parser.add_argument("--action-ckpt", type=Path, default=DEFAULT_ACTION)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--ffprobe", default=os.environ.get("QVG_FFPROBE", "ffprobe"))
    parser.add_argument("--codebook-root", type=Path)
    parser.add_argument("--routing-path", type=Path)
    parser.add_argument("--tree-path", type=Path)
    parser.add_argument("--v-params-path", type=Path)
    parser.add_argument("--profile-path", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--frames", type=int, default=157)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--record-latent-hashes", action="store_true")
    parser.add_argument("--append-only-vision-cache", action="store_true")
    parser.add_argument("--gpu-resident-compressed-kv", action="store_true")
    parser.add_argument("--terminal-cache-audit", action="store_true")
    parser.add_argument(
        "--segmented-decode-backend",
        choices=("legacy", "preallocated"),
        default="legacy",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model_path is None:
        raise ValueError("--model-path or QVG_HY8B_MODEL is required")
    if args.action_ckpt is None:
        raise ValueError("--action-ckpt or QVG_HY_ACTION_CKPT is required")
    if args.output_dir.exists():
        raise FileExistsError(f"create-only output exists: {args.output_dir}")
    initial_gpu = query_gpu(args.gpu)
    initial_pids = compute_pids(args.gpu)
    if initial_gpu["used_mib"] > 512 or initial_gpu["utilization"] > 5 or initial_pids:
        raise RuntimeError(f"GPU {args.gpu} is not exclusive: {initial_gpu}, pids={sorted(initial_pids)}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    row = record_for(manifest, args.split, args.prompt_index)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": "hyworldplay_moviegen_i2v_job.v1",
        "method": args.method,
        "codec": METHOD_CODEC[args.method],
        "split": args.split,
        "prompt_index": args.prompt_index,
        "dataset_row": row["dataset_row"],
        "prompt": row["prompt"],
        "conditioning_image_path": row["conditioning_image_path"],
        "conditioning_image_sha256": row["conditioning_image_sha256"],
        "seed": args.seed,
        "initial_noise_identity": "torch.Generator(CUDA).manual_seed(seed) in a fresh process",
        "frames": args.frames,
        "chunks": (args.frames + 3) // 16,
        "width": 848,
        "height": 464,
        "fps": 24,
        "steps": 4,
        "dtype": "bf16",
        "cfg": 1,
        "pose": f"w-{(args.frames - 1) // 4}",
        "full_history_attention": True,
        "append_only_vision_cache": args.append_only_vision_cache,
        "gpu_resident_compressed_kv": args.gpu_resident_compressed_kv,
        "segmented_decode_backend": args.segmented_decode_backend,
        "kv_cpu_offload": True,
        "component_offload": True,
        "sr": False,
        "rewrite": False,
        "terminal_cache_audit": bool(args.terminal_cache_audit or args.method == "collect"),
        "end_to_end_accounting": (
            "process start through finalized MP4 mtime; terminal audit runs "
            "after MP4 saving and is excluded"
        ),
        "rvq_relative_stop_fast_approximate": os.getenv(
            "HYWORLDPLAY_RVQ_RELATIVE_STOP_FAST_APPROXIMATE", "0"
        ),
        "rvq_relative_stop_monolithic_exact": os.getenv(
            "HYWORLDPLAY_RVQ_RELATIVE_STOP_MONOLITHIC_EXACT", "0"
        ),
        "cache_decode_prefetch": os.getenv(
            "QVG_HY_CACHE_DECODE_PREFETCH", "0"
        ),
        "rvq_monolithic_exact_audit": os.getenv(
            "HYWORLDPLAY_RVQ_MONOLITHIC_EXACT_AUDIT", "0"
        ),
        "rvq_staged_defer_stop_checks": os.getenv(
            "HYWORLDPLAY_RVQ_STAGED_DEFER_STOP_CHECKS", "0"
        ),
        "rvq_staged_batch2": os.getenv(
            "HYWORLDPLAY_RVQ_STAGED_BATCH2", "0"
        ),
        "rvq_staged_stop_check_stride": os.getenv(
            "HYWORLDPLAY_RVQ_STAGED_STOP_CHECK_STRIDE", "2"
        ),
    }
    write_create(args.output_dir / "config.json", config)
    rvq_codebooks = (
        codebook_storage_metadata(args.codebook_root)
        if args.method in OURS_METHODS
        else None
    )
    run_info = {
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "dirty_state": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"],
            text=True,
        ).splitlines(),
        "config_path": str((args.output_dir / "config.json").resolve()),
        "config_hash": sha256_file(args.output_dir / "config.json"),
        "codebook_path": str(args.codebook_root.resolve()) if args.codebook_root else None,
        "codebook_hash": sha256_file(args.codebook_root / "manifest.json") if args.codebook_root and (args.codebook_root / "manifest.json").is_file() else None,
        "model_path": str(args.model_path.resolve()),
        "model_hash": args.model_hash,
        "action_checkpoint_path": str(args.action_ckpt.resolve()),
        "action_checkpoint_hash": sha256_file(args.action_ckpt),
        "hostname": socket.gethostname(),
        "gpu_type": initial_gpu["name"],
        "gpu": initial_gpu,
        "seed": args.seed,
        "prompt_list": [row["prompt"]],
        "conditioning_image_path": row["conditioning_image_path"],
        "conditioning_image_sha256": row["conditioning_image_sha256"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_hash": sha256_file(args.manifest),
        "timestamp": now(),
        "codec_artifacts": {
            name: (
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                if path is not None and path.is_file()
                else None
            )
            for name, path in {
                "routing": args.routing_path,
                "tree": args.tree_path,
                "v_params": args.v_params_path,
                "profile": args.profile_path,
            }.items()
        },
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (
                REPO_ROOT / "quant_videogen/hyworldplay8b_cache_codec.py",
                REPO_ROOT / "quant_videogen/compress.py",
                REPO_ROOT / "quant_videogen/kv_shared_budget.py",
                REPO_ROOT / "quant_videogen/hyworldplay_capture.py",
                REPO_ROOT / "quant_videogen/hyworldplay_residual_stats.py",
                REPO_ROOT / "quant_videogen/hyworldplay_attention_calibration.py",
                REPO_ROOT / "quant_videogen/hyworldplay_append_only.py",
                REPO_ROOT / "quant_videogen/rvq.py",
                REPO_ROOT / "quant_videogen/kernels/triton/rvq.py",
                REPO_ROOT / "quant_videogen/sim/quant/quantize_config.py",
                REPO_ROOT / "quant_videogen/temporal_adaptive_residual.py",
                REPO_ROOT / "quant_videogen/uncompress.py",
                REPO_ROOT / "experiments/HY-WorldPlay/hyvideo/commons/infer_state.py",
                REPO_ROOT / "experiments/HY-WorldPlay/hyvideo/models/transformers/worldplay_1_5_transformer.py",
                REPO_ROOT / "experiments/HY-WorldPlay/hyvideo/models/transformers/modules/attention.py",
                REPO_ROOT / "experiments/HY-WorldPlay/hyvideo/pipelines/worldplay_video_pipeline.py",
                REPO_ROOT / "experiments/HY-WorldPlay/hyvideo/generate.py",
                Path(__file__).resolve(),
            )
        },
        "rvq_codebook_storage": rvq_codebooks,
    }
    write_create(args.output_dir / "run_info.json", run_info)
    env = os.environ.copy()
    env.update({
        "PYTHON": str(args.python),
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "MODEL_PATH": str(args.model_path),
        "ACTION_CKPT": str(args.action_ckpt),
        "IMAGE_PATH": row["conditioning_image_path"],
        "OUTPUT_PATH": str(args.output_dir),
        "PROMPT": row["prompt"],
        "POSE": config["pose"],
        "VIDEO_LENGTH": str(args.frames),
        "SEED": str(args.seed),
        "NUM_INFERENCE_STEPS": "4",
        "HEIGHT": "464",
        "WIDTH": "848",
        "OFFLOADING": "true",
        "KV_CACHE_OFFLOAD": "true",
        "KV_CACHE_GPU_RESIDENT_COMPRESSED": (
            "true" if args.gpu_resident_compressed_kv else "false"
        ),
        "FULL_HISTORY_ATTENTION": "true",
        "APPEND_ONLY_VISION_CACHE": "true" if args.append_only_vision_cache else "false",
        "KV_CACHE_CODEC": METHOD_CODEC[args.method],
        "GROUP_OFFLOADING": "false",
        "SR": "false",
        "REWRITE": "false",
        "FEW_STEP": "true",
        "MASTER_PORT": str(31000 + args.gpu * 100 + (os.getpid() % 97)),
        "QVG_CODEC_BREAKDOWN": "1",
        "QVG_HY_SEGMENTED_DECODE_BACKEND": args.segmented_decode_backend,
        "RECORD_LATENT_HASHES": "true" if args.record_latent_hashes else "false",
        "TERMINAL_CACHE_AUDIT": "true" if args.terminal_cache_audit else "false",
    })
    if args.method == "collect":
        if args.capture_root is None:
            raise ValueError("collect requires --capture-root")
        env.update({
            "KV_CAPTURE_DIR": str(args.capture_root),
            "KV_CAPTURE_PROMPT_INDEX": str(args.prompt_index),
            "KV_CAPTURE_SAMPLE_TOKENS": "1250",
            "KV_CAPTURE_SEED": "42",
            "KV_CAPTURE_HISTORY_CALLS": str(config["chunks"]),
            # The final completed span is built after MP4 saving and is not
            # part of generation E2E.
            "TERMINAL_CACHE_AUDIT": "true",
        })
    if args.method == "collect_residual":
        if args.codebook_root is None:
            raise ValueError("collect_residual requires --codebook-root")
        env.update({
            "KV_CODEC_K_CODEBOOK_DIR": str(args.codebook_root / "nonorm_k_l16c64_codebooks"),
            "KV_CODEC_V_CODEBOOK_DIR": str(args.codebook_root / "nonorm_v_l16c64_codebooks"),
            "KV_CODEC_FRAME_SPATIAL_TOKENS": "1537",
            "KV_RESIDUAL_STATS_DIR": str(args.capture_root or args.output_dir.parent),
            "KV_RESIDUAL_STATS_PROMPT_INDEX": str(args.prompt_index),
            "KV_RESIDUAL_STATS_SAMPLES_PER_HEAD": "128",
            "KV_RESIDUAL_STATS_SEED": "42",
        })
    if args.method == "calibrate_avg2":
        required = [args.codebook_root, args.routing_path, args.tree_path, args.v_params_path]
        if any(value is None for value in required):
            raise ValueError(
                "calibrate_avg2 requires codebooks, routing, tree, and V params"
            )
        env.update({
            "KV_CODEC_K_CODEBOOK_DIR": str(args.codebook_root / "nonorm_k_l16c64_codebooks"),
            "KV_CODEC_V_CODEBOOK_DIR": str(args.codebook_root / "nonorm_v_l16c64_codebooks"),
            "KV_CODEC_ROUTING_PATH": str(args.routing_path),
            "KV_CODEC_TREE_PATH": str(args.tree_path),
            "KV_CODEC_V_PARAMS_PATH": str(args.v_params_path),
            "KV_CODEC_FRAME_SPATIAL_TOKENS": "1537",
            "KV_AVG2_CALIBRATION_DIR": str(args.capture_root or args.output_dir.parent),
            "KV_AVG2_CALIBRATION_PROMPT_INDEX": str(args.prompt_index),
            "KV_AVG2_CALIBRATION_QUERIES_PER_STREAM": "4",
            "KV_AVG2_CALIBRATION_SEED": "42",
        })
    if args.method in OURS_METHODS:
        required = [args.codebook_root, args.routing_path, args.tree_path, args.v_params_path, args.profile_path]
        if any(value is None for value in required):
            raise ValueError(f"{args.method} requires codebooks, routing, tree, V params, and temporal profile")
        env.update({
            "KV_CODEC_K_CODEBOOK_DIR": str(args.codebook_root / "nonorm_k_l16c64_codebooks"),
            "KV_CODEC_V_CODEBOOK_DIR": str(args.codebook_root / "nonorm_v_l16c64_codebooks"),
            "KV_CODEC_ROUTING_PATH": str(args.routing_path),
            "KV_CODEC_TREE_PATH": str(args.tree_path),
            "KV_CODEC_V_PARAMS_PATH": str(args.v_params_path),
            "KV_CODEC_TEMPORAL_PROFILE_PATH": str(args.profile_path),
            "KV_CODEC_FRAME_SPATIAL_TOKENS": "1537",
        })
    command = ["bash", str(REPO_ROOT / "scripts/HY-WorldPlay/run_hunyuan8b_bf16.sh"), "--metrics_path", str(args.output_dir / "generation_metrics.json")]
    monitor = NvmlMonitor(args.gpu, initial_pids)
    started = time.perf_counter()
    started_epoch = time.time()
    with (args.output_dir / "generation.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor.allow_process_tree(process.pid)
        monitor.start()
        returncode = process.wait()
    monitor.stop()
    elapsed = time.perf_counter() - started
    status = "success" if returncode == 0 else "failed"
    video_info = None
    metrics = None
    error = None
    try:
        if status == "success":
            video_info = probe(args.output_dir / "gen.mp4", args.ffprobe)
            if video_info != {"width": 848, "height": 464, "fps": 24.0, "frames": args.frames}:
                raise RuntimeError(f"unexpected video properties: {video_info}")
            metrics = json.loads((args.output_dir / "generation_metrics.json").read_text(encoding="utf-8"))
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    if monitor.external_pids:
        status = "contaminated"
        error = f"external GPU compute PIDs observed: {sorted(monitor.external_pids)}"
    mp4_e2e_seconds = None
    if video_info is not None:
        mp4_e2e_seconds = max(
            0.0, (args.output_dir / "gen.mp4").stat().st_mtime - started_epoch
        )
    generation_end_to_end_seconds = (
        mp4_e2e_seconds if mp4_e2e_seconds is not None else elapsed
    )
    result = {
        "schema_version": "hyworldplay_moviegen_i2v_result.v1",
        "status": status,
        "returncode": returncode,
        "error": error,
        "end_to_end_seconds": generation_end_to_end_seconds,
        "end_to_end_includes_terminal_audit": False,
        "generation_end_to_end_excluding_terminal_audit_seconds": generation_end_to_end_seconds,
        "process_wall_seconds": elapsed,
        "nvml_peak_used_mib": monitor.peak_mib,
        "nvml_samples": monitor.samples,
        "observed_compute_pids": sorted(monitor.observed_pids),
        "external_compute_pids": sorted(monitor.external_pids),
        "video": video_info,
        "encode_e2e": summarize_events(metrics.get("kv_codec_encode_events", [])) if metrics else None,
        "decode_e2e": summarize_events(metrics.get("kv_codec_decode_events", [])) if metrics else None,
        "part_timing_seconds": metrics.get("part_timing_seconds") if metrics else None,
        "chunk_timing_events": metrics.get("chunk_timing_events") if metrics else None,
        "codec_encode_cuda_breakdown": metrics.get("codec_encode_cuda_breakdown") if metrics else None,
        "codec_decode_cuda_breakdown": metrics.get("codec_decode_cuda_breakdown") if metrics else None,
        "kv_cache_gpu_compressed_peak_mib": metrics.get("kv_cache_gpu_compressed_peak_mib") if metrics else None,
        "kv_cache_gpu_resident_hits": metrics.get("kv_cache_gpu_resident_hits") if metrics else None,
        "kv_cache_gpu_resident_promotions": metrics.get("kv_cache_gpu_resident_promotions") if metrics else None,
        "top_level_unattributed_seconds": metrics.get("top_level_unattributed_seconds") if metrics else None,
        "terminal_cache_audit": metrics.get("terminal_cache_audit") if metrics else None,
        "rvq_temporal_physical_accounting": (
            summarize_avg2_accounting(metrics.get("kv_codec_encode_events", []), rvq_codebooks)
            if metrics else None
        ),
        # Compatibility alias retained for existing avg2 report readers.
        "rvq_avg2_physical_accounting": (
            summarize_avg2_accounting(metrics.get("kv_codec_encode_events", []), rvq_codebooks)
            if metrics else None
        ),
        "generation_metrics_path": str((args.output_dir / "generation_metrics.json").resolve()) if metrics else None,
        "completed_at": now(),
    }
    write_create(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
