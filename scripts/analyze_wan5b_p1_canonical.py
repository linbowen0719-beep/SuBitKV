#!/usr/bin/env python3
"""P1 test of train-fitted low-cost canonicalization for Wan2.2 5B RVQ sharing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402
import analyze_wan5b_p1_compact as compact  # noqa: E402


LAYERS = p0.LAYERS
HEADS = p0.HEADS
DIM = p0.DIM
GROUPS = LAYERS * HEADS


def output_root(cfg: dict[str, Any]) -> Path:
    return p0.resolve_path(cfg["output_dir"])


def parent_root(cfg: dict[str, Any]) -> Path:
    return p0.resolve_path(cfg["p0_result_dir"])


def task_list(cfg: dict[str, Any]) -> list[tuple[int, int, str]]:
    return [
        (int(fold), int(seed), str(tensor_name))
        for fold in cfg["task_folds"]
        for seed in cfg["initialization_seeds"]
        for tensor_name in cfg["tensors"]
    ]


def collect_group_matrix(
    cfg: dict[str, Any], tensor_name: str, prompt_ids: Sequence[int], part: str
) -> torch.Tensor:
    layers, heads = compact.axes(cfg)
    if layers != list(range(LAYERS)) or heads != list(range(HEADS)):
        raise ValueError("Canonical sharing requires all 720 layer-head groups")
    return torch.cat(
        [
            compact.collect_samples(cfg, tensor_name, layer, prompt_ids, part)
            for layer in layers
        ],
        dim=0,
    ).contiguous()


def fit_transform(train: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    values = train.float()
    if name == "center":
        offset = values.mean(dim=1, keepdim=True)
        scale = torch.ones((values.shape[0], 1, 1), dtype=torch.float32)
    elif name == "scalar_rms":
        offset = torch.zeros((values.shape[0], 1, 1), dtype=torch.float32)
        scale = torch.sqrt((values * values).mean(dim=(1, 2), keepdim=True)).clamp_min(1e-6)
    elif name == "channel_zscore":
        offset = values.mean(dim=1, keepdim=True)
        scale = values.std(dim=1, keepdim=True).clamp_min(1e-6)
    else:
        raise ValueError(f"Unknown canonicalization {name}")
    return offset, scale


def apply_transform(x: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() - offset) / scale


def invert_transform(x: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale + offset


def metadata_values_per_group(transform: str) -> int:
    if transform == "raw":
        return 0
    if transform == "center":
        return DIM
    if transform == "scalar_rms":
        return 1
    if transform == "channel_zscore":
        return 2 * DIM
    raise ValueError(transform)


def sharing_rate(cfg: dict[str, Any], transform: str, count: int) -> dict[str, Any]:
    levels = int(cfg["rvq"]["levels"])
    base = p0.codebook_rate(cfg, levels)
    metadata_bytes = metadata_values_per_group(transform) * 2
    codebook_total = float(base["codebook_storage_bytes_per_group"]) * int(count)
    amortization = int(cfg["rvq"]["amortization_vectors_per_group"])
    logical_bits = float(base["logical_index_bits_per_vector"])
    total = logical_bits + (
        codebook_total / GROUPS + metadata_bytes
    ) * 8 / amortization
    return {
        "logical_index_bits_per_vector": logical_bits,
        "total_codebook_count": int(count),
        "local_group_count": GROUPS,
        "codebook_storage_bytes_total": codebook_total,
        "codebook_storage_mib_total": codebook_total / 1024**2,
        "transform_metadata_bytes_per_group": metadata_bytes,
        "transform_metadata_mib_per_tensor": metadata_bytes * GROUPS / 1024**2,
        "total_amortized_bits_per_vector": total,
        "total_amortized_bits_per_scalar": total / DIM,
    }


def local_codebooks(
    cfg: dict[str, Any], fold: int, seed: int, tensor_name: str
) -> torch.Tensor:
    result = []
    for layer in range(LAYERS):
        payload = torch.load(
            compact.p0_codebook_path(cfg, fold, seed, tensor_name, layer),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        result.append(payload["codebooks"])
    return torch.cat(result, dim=0).contiguous()


def raw_assignment_path(cfg: dict[str, Any], fold: int, seed: int, tensor_name: str) -> Path:
    return (
        parent_root(cfg)
        / "h4"
        / f"fold_{fold}"
        / f"seed_{seed}"
        / tensor_name.lower()
        / "sharing_assignments.json"
    )


def raw_scaling_path(cfg: dict[str, Any], fold: int, seed: int, tensor_name: str) -> Path:
    return raw_assignment_path(cfg, fold, seed, tensor_name).with_name(
        "codebook_count_scaling.csv"
    )


def labels_for_count(
    cfg: dict[str, Any], fold: int, seed: int, tensor_name: str, count: int
) -> torch.Tensor:
    record = json.loads(
        raw_assignment_path(cfg, fold, seed, tensor_name).read_text(encoding="utf-8")
    )
    values = record["assignments"][f"feature_kmeans_{int(count)}"]
    labels = torch.tensor(values, dtype=torch.long)
    if labels.shape != (GROUPS,) or torch.unique(labels).numel() != int(count):
        raise ValueError(f"Invalid P0 labels for count={count}")
    return labels


def fit_shared(
    cfg: dict[str, Any], train: torch.Tensor, labels: torch.Tensor, count: int,
    seed: int, device: torch.device,
) -> torch.Tensor:
    fit_cfg = dict(cfg)
    fit_cfg["h4"] = {"shared_train_samples": int(cfg["h5"]["shared_train_samples"])}
    return p0.fit_shared_codebooks(fit_cfg, train, labels, count, seed, device)


def unit_paths(root: Path, fold: int, seed: int, tensor_name: str) -> dict[str, Path]:
    base = root / "canonical" / f"fold_{fold}" / f"seed_{seed}" / tensor_name.lower()
    return {
        "codebook": base / "canonical_codebooks.pt",
        "metrics": base / "sharing_metrics.csv",
        "done": base / "done.json",
    }


def compute_local_baseline(
    cfg: dict[str, Any], tensor_name: str, test_ids: Sequence[int],
    codebooks_cpu: torch.Tensor, device: torch.device,
) -> dict[tuple[int, int, int], float]:
    result: dict[tuple[int, int, int], float] = {}
    for prompt_id in test_ids:
        x_cpu = collect_group_matrix(cfg, tensor_name, [prompt_id], "eval")
        for start in range(0, GROUPS, 24):
            end = min(GROUPS, start + 24)
            x = x_cpu[start:end].to(device=device, dtype=torch.float32)
            cb = codebooks_cpu[start:end].to(device=device, dtype=torch.float32)
            reconstruction = p0.decode_grouped_rvq(x, cb)
            metrics = p0.reconstruction_metrics(x, reconstruction)
            for local, group in enumerate(range(start, end)):
                result[(int(prompt_id), group // HEADS, group % HEADS)] = float(
                    metrics["relative_mse"][local].item()
                )
    return result


def raw_rows(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    local: dict[tuple[int, int, int], float],
) -> list[dict[str, Any]]:
    counts = {int(value) for value in cfg["h5"]["codebook_counts"]}
    result = []
    with raw_scaling_path(cfg, fold, seed, tensor_name).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            count = int(row["total_codebook_count"])
            if row["grouping_policy"] != "feature_kmeans" or count not in counts:
                continue
            prompt_id = int(row["prompt_id"])
            layer = int(row["layer"])
            head = int(row["head"])
            baseline = local[(prompt_id, layer, head)]
            relative_mse = float(row["relative_mse"])
            result.append(
                {
                    **row,
                    "study_id": cfg["study_id"],
                    "config_hash": cfg_hash,
                    "hypothesis_id": "P1-C1",
                    "transform": "raw",
                    "grouping_policy": cfg["h5"]["grouping_policy"],
                    "local_relative_mse": baseline,
                    "distortion_to_local_ratio": relative_mse / max(baseline, 1e-20),
                    **sharing_rate(cfg, "raw", count),
                }
            )
    return result


def evaluate_canonical_count(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    transform: str, count: int, labels: torch.Tensor, codebooks: torch.Tensor,
    offset: torch.Tensor, scale: torch.Tensor, test_ids: Sequence[int],
    local: dict[tuple[int, int, int], float], device: torch.device,
) -> list[dict[str, Any]]:
    result = []
    group_codebooks = codebooks[labels].contiguous()
    rate = sharing_rate(cfg, transform, count)
    for prompt_id in test_ids:
        x_cpu = collect_group_matrix(cfg, tensor_name, [prompt_id], "eval")
        for start in range(0, GROUPS, 24):
            end = min(GROUPS, start + 24)
            original = x_cpu[start:end].to(device=device, dtype=torch.float32)
            local_offset = offset[start:end].to(device=device)
            local_scale = scale[start:end].to(device=device)
            canonical = apply_transform(original, local_offset, local_scale)
            cb = group_codebooks[start:end].to(device=device, dtype=torch.float32)
            canonical_reconstruction = p0.decode_grouped_rvq(canonical, cb)
            reconstruction = invert_transform(
                canonical_reconstruction, local_offset, local_scale
            )
            metrics = p0.reconstruction_metrics(original, reconstruction)
            for local_index, group in enumerate(range(start, end)):
                layer, head = divmod(group, HEADS)
                baseline = local[(int(prompt_id), layer, head)]
                relative_mse = float(metrics["relative_mse"][local_index].item())
                result.append(
                    {
                        "study_id": cfg["study_id"],
                        "config_hash": cfg_hash,
                        "hypothesis_id": "P1-C1",
                        "fold": fold,
                        "initialization_seed": seed,
                        "prompt_id": int(prompt_id),
                        "layer": layer,
                        "head": head,
                        "K_or_V": tensor_name,
                        "hook_stage": "pre_rope" if tensor_name == "K" else "pre_prope",
                        "method": "canonical_shared_rvq",
                        "transform": transform,
                        "grouping_policy": cfg["h5"]["grouping_policy"],
                        "sharing_cluster": int(labels[group].item()),
                        "local_relative_mse": baseline,
                        "distortion_to_local_ratio": relative_mse / max(baseline, 1e-20),
                        **rate,
                        **{
                            key: float(value[local_index].item())
                            for key, value in metrics.items()
                        },
                    }
                )
    return result


def run_unit(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int,
    tensor_name: str, device: torch.device,
) -> None:
    paths = unit_paths(output_root(cfg), fold, seed, tensor_name)
    if all(path.exists() for path in paths.values()):
        return
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite incomplete canonical unit: {existing}")
    started = time.time()
    train_ids, test_ids = compact.fold_ids(cfg, fold)
    train = collect_group_matrix(cfg, tensor_name, train_ids, "train")
    local_cb = local_codebooks(cfg, fold, seed, tensor_name)
    local = compute_local_baseline(cfg, tensor_name, test_ids, local_cb, device)
    rows = raw_rows(cfg, cfg_hash, fold, seed, tensor_name, local)
    saved_codebooks: dict[str, dict[int, torch.Tensor]] = {}
    saved_parameters: dict[str, dict[str, torch.Tensor]] = {}
    for transform_index, transform in enumerate(cfg["h5"]["transforms"]):
        transform = str(transform)
        offset, scale = fit_transform(train, transform)
        canonical_train = apply_transform(train, offset, scale)
        saved_codebooks[transform] = {}
        saved_parameters[transform] = {
            "offset": offset.to(torch.bfloat16),
            "scale": scale.to(torch.bfloat16),
        }
        for count in cfg["h5"]["codebook_counts"]:
            count = int(count)
            labels = labels_for_count(cfg, fold, seed, tensor_name, count)
            fit_seed = (
                seed
                + fold * 1_000_003
                + count * 1009
                + transform_index * 100_000_007
                + (0 if tensor_name == "K" else 500_009)
            )
            codebooks = fit_shared(
                cfg, canonical_train, labels, count, fit_seed, device
            )
            rows.extend(
                evaluate_canonical_count(
                    cfg, cfg_hash, fold, seed, tensor_name, transform, count,
                    labels, codebooks, offset, scale, test_ids, local, device,
                )
            )
            saved_codebooks[transform][count] = codebooks.to(torch.bfloat16)
        del canonical_train
    payload = {
        "schema_version": "wan5b_p1_canonical_sharing.v1",
        "config_hash": cfg_hash,
        "fold": fold,
        "initialization_seed": seed,
        "tensor": tensor_name,
        "train_prompt_ids": train_ids,
        "test_prompt_ids": test_ids,
        "grouping_policy": cfg["h5"]["grouping_policy"],
        "codebook_counts": [int(value) for value in cfg["h5"]["codebook_counts"]],
        "canonical_parameters": saved_parameters,
        "codebooks": saved_codebooks,
    }
    p0.atomic_torch(paths["codebook"], payload)
    p0.write_csv(paths["metrics"], rows)
    p0.atomic_json(
        paths["done"],
        {
            "completed": p0.utc_now(),
            "config_hash": cfg_hash,
            "fold": fold,
            "initialization_seed": seed,
            "tensor": tensor_name,
            "rows": len(rows),
            "codebook_sha256": p0.sha256_file(paths["codebook"]),
            "seconds": time.time() - started,
        },
    )
    print(
        f"[canonical] fold={fold} seed={seed} tensor={tensor_name} "
        f"rows={len(rows)} seconds={time.time()-started:.1f}",
        flush=True,
    )


def validate_inputs(cfg: dict[str, Any]) -> None:
    layers, heads = compact.axes(cfg)
    if layers != list(range(LAYERS)) or heads != list(range(HEADS)):
        raise ValueError("P1-C1 requires all layers and heads")
    allowed = {"center", "scalar_rms", "channel_zscore"}
    if not set(cfg["h5"]["transforms"]).issubset(allowed):
        raise ValueError("Unregistered canonical transform")
    for fold, seed, tensor_name in task_list(cfg):
        for count in cfg["h5"]["codebook_counts"]:
            labels_for_count(cfg, fold, seed, tensor_name, int(count))
        if not raw_scaling_path(cfg, fold, seed, tensor_name).exists():
            raise FileNotFoundError(raw_scaling_path(cfg, fold, seed, tensor_name))
        for layer in range(LAYERS):
            path = compact.p0_codebook_path(cfg, fold, seed, tensor_name, layer)
            if not path.exists():
                raise FileNotFoundError(path)


def cmd_prepare(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    validate_inputs(cfg)
    root = output_root(cfg)
    info_path = root / "run_info.json"
    if root.exists() and not info_path.exists():
        raise FileExistsError(f"Refusing existing output without run_info: {root}")
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("config_hash") != cfg_hash:
            raise ValueError("Existing config hash mismatch")
        print(f"[prepare] already prepared root={root}")
        return
    items = p0.prompt_items(Path(cfg["data_root"]))
    parent_manifest = parent_root(cfg) / "cv" / "codebook_manifest.json"
    registry = REPO_ROOT / "artifacts" / "p1_compact_hypotheses.csv"
    info = {
        "git_commit": p0.git_commit(),
        "config_path": str(cfg_path),
        "config_hash": cfg_hash,
        "codebook_path": {"p0_local_and_raw_sharing": str(parent_root(cfg).resolve())},
        "codebook_hash": {"p0_cv_manifest": p0.sha256_file(parent_manifest)},
        "hostname": p0.socket.gethostname(),
        "gpu_type": p0.gpu_name(),
        "seed": int(cfg["seed"]),
        "initialization_seeds": [int(value) for value in cfg["initialization_seeds"]],
        "prompt_list": [str(item["prompt"]) for item in items],
        "timestamp": p0.utc_now(),
        "model_id": cfg["model_id"],
        "source_sample_cache": str((parent_root(cfg) / "sample_cache").resolve()),
        "source_data_read_only": True,
        "statistical_unit": "prompt",
        "folds": p0.make_folds(list(range(20)), int(cfg["folds"]), int(cfg["seed"])),
        "hypothesis_registry": str(registry.resolve()),
        "hypothesis_registry_hash": p0.sha256_file(registry),
        "transform_fit_scope": "16 training prompts in each fold only",
        "grouping_control": "P0 raw feature-kmeans assignments fixed across transforms",
    }
    p0.atomic_json(info_path, info)
    p0.write_csv(
        root / "tables" / "rate_accounting.csv",
        [
            sharing_rate(cfg, transform, int(count))
            | {"transform": transform}
            for transform in ["raw", *cfg["h5"]["transforms"]]
            for count in cfg["h5"]["codebook_counts"]
        ],
    )
    print(f"[prepare] root={root} config_hash={cfg_hash}", flush=True)


def cmd_worker(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // int(args.num_shards)))
    torch.set_num_interop_threads(1)
    selected = [
        task
        for index, task in enumerate(task_list(cfg))
        if index % int(args.num_shards) == int(args.shard_id)
    ]
    print(
        f"[worker] shard={args.shard_id}/{args.num_shards} tasks={len(selected)} "
        f"gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    for task in selected:
        run_unit(cfg, cfg_hash, *task, torch.device("cuda:0"))


def cmd_launch(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, _cfg_hash = p0.parse_config(cfg_path)
    gpus = [int(value) for value in (args.gpus.split(",") if args.gpus else cfg["gpus"])]
    if not 1 <= len(gpus) <= 4 or len(set(gpus)) != len(gpus):
        raise ValueError(f"Expected one to four distinct GPUs, got {gpus}")
    launch_id = p0.utc_now().replace(":", "").replace("+", "_")
    log_dir = output_root(cfg) / "logs" / f"canonical_{launch_id}"
    log_dir.mkdir(parents=True, exist_ok=False)
    processes = []
    for shard, gpu in enumerate(gpus):
        path = log_dir / f"gpu_{gpu}_shard_{shard}.log"
        handle = path.open("x", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            str(cfg["python"]), str(Path(__file__).resolve()), "worker",
            "--config", str(cfg_path), "--shard-id", str(shard),
            "--num-shards", str(len(gpus)),
        ]
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        processes.append((gpu, process, handle, path))
        print(f"[launch] gpu={gpu} pid={process.pid} log={path}", flush=True)
    last_report = 0.0
    try:
        while any(process.poll() is None for _gpu, process, _handle, _path in processes):
            if time.time() - last_report >= 10:
                print(
                    "[monitor] "
                    + ", ".join(
                        f"gpu{gpu}:pid{process.pid}:"
                        f"{'running' if process.poll() is None else process.returncode}"
                        for gpu, process, _handle, _path in processes
                    ),
                    flush=True,
                )
                last_report = time.time()
            time.sleep(2)
    finally:
        for _gpu, _process, handle, _path in processes:
            handle.close()
    failed = [item for item in processes if item[1].returncode != 0]
    if failed:
        for gpu, process, _handle, path in failed:
            print(
                f"[failed] gpu={gpu} rc={process.returncode} log={path}\n"
                f"{path.read_text(encoding='utf-8')[-12000:]}",
                flush=True,
            )
        raise RuntimeError(f"{len(failed)} worker(s) failed; logs preserved at {log_dir}")
    print(f"[launch] all workers complete logs={log_dir}", flush=True)


def cmd_merge(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    root = output_root(cfg)
    expected = len(task_list(cfg))
    done = list((root / "canonical").glob("fold_*/seed_*/*/done.json"))
    if len(done) != expected:
        raise RuntimeError(f"Incomplete canonical run: {len(done)}/{expected}")
    rows = p0.merge_csv_glob(
        (root / "canonical").glob("fold_*/seed_*/*/sharing_metrics.csv"),
        root / "tables" / "canonical_sharing_metrics.csv",
    )
    files = []
    for path in sorted((root / "canonical").glob("fold_*/seed_*/*/canonical_codebooks.pt")):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": p0.sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "wan5b_p1_canonical_manifest.v1",
        "config_hash": cfg_hash,
        "files": files,
        "total_bytes": sum(int(row["bytes"]) for row in files),
    }
    manifest_path = root / "canonical" / "codebook_manifest.json"
    p0.atomic_json(manifest_path, manifest)
    info_path = root / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["output_codebook_path"] = str((root / "canonical").resolve())
    info["output_codebook_hash"] = p0.sha256_file(manifest_path)
    p0.atomic_json(info_path, info)
    p0.atomic_json(
        root / "canonical" / "canonical.done.json",
        {
            "completed": p0.utc_now(),
            "config_hash": cfg_hash,
            "units": expected,
            "rows": rows,
            "codebook_files": len(files),
            "codebook_bytes": manifest["total_bytes"],
        },
    )
    print(f"[merge] units={expected} rows={rows}", flush=True)


def stream_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def cmd_summarize(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    root = output_root(cfg)
    metric_path = root / "tables" / "canonical_sharing_metrics.csv"
    sums: dict[tuple[str, int, str, int], list[float]] = {}
    for row in stream_rows(metric_path):
        key = (
            row["transform"], int(row["total_codebook_count"]),
            row["K_or_V"], int(row["prompt_id"]),
        )
        value = sums.setdefault(key, [0.0, 0.0, 0.0])
        value[0] += float(row["relative_mse"])
        value[1] += float(row["local_relative_mse"])
        value[2] += 1
    prompt_rows = []
    lookup: dict[tuple[str, int, str, int], float] = {}
    for key, (distortion, local, count_rows) in sorted(sums.items()):
        ratio = distortion / max(local, 1e-20)
        lookup[key] = ratio
        prompt_rows.append(
            {
                "transform": key[0], "total_codebook_count": key[1],
                "K_or_V": key[2], "prompt_id": key[3],
                "mean_relative_mse": distortion / count_rows,
                "mean_local_relative_mse": local / count_rows,
                "distortion_to_local_ratio": ratio,
                "group_seed_rows": int(count_rows),
            }
        )
    bootstrap_samples = int(cfg["bootstrap"]["samples"])
    bootstrap_seed = int(cfg["bootstrap"]["seed"])
    prompts = sorted({int(row["prompt_id"]) for row in prompt_rows})
    summary_rows = []
    for transform_index, transform in enumerate(["raw", *cfg["h5"]["transforms"]]):
        for count_index, count in enumerate(cfg["h5"]["codebook_counts"]):
            for tensor_index, tensor_name in enumerate(("K", "V")):
                ratios = [
                    lookup[(str(transform), int(count), tensor_name, prompt_id)]
                    for prompt_id in prompts
                ]
                ci_low, ci_high = compact.bootstrap_ci(
                    ratios, bootstrap_samples,
                    bootstrap_seed + transform_index * 1009 + count_index * 31 + tensor_index,
                )
                raw_ratios = [
                    lookup[("raw", int(count), tensor_name, prompt_id)]
                    for prompt_id in prompts
                ]
                paired_improvements = [
                    (raw - candidate) / max(raw, 1e-20)
                    for raw, candidate in zip(raw_ratios, ratios)
                ]
                imp_low, imp_high = compact.bootstrap_ci(
                    paired_improvements, bootstrap_samples,
                    bootstrap_seed + 50_000 + transform_index * 1009 + count_index * 31 + tensor_index,
                )
                summary_rows.append(
                    {
                        "transform": transform,
                        "total_codebook_count": int(count),
                        "K_or_V": tensor_name,
                        "prompts": len(prompts),
                        "mean_distortion_to_local_ratio": float(np.mean(ratios)),
                        "median_distortion_to_local_ratio": float(np.median(ratios)),
                        "bootstrap_95_ci_low": ci_low,
                        "bootstrap_95_ci_high": ci_high,
                        "mean_improvement_vs_raw": float(np.mean(paired_improvements)),
                        "improvement_vs_raw_95_ci_low": imp_low,
                        "improvement_vs_raw_95_ci_high": imp_high,
                    }
                )
    p0.write_csv(root / "tables" / "prompt_aggregate.csv", prompt_rows)
    p0.write_csv(root / "tables" / "canonical_summary.csv", summary_rows)
    supported_candidates = []
    for transform in cfg["h5"]["transforms"]:
        for count in cfg["h5"]["codebook_counts"]:
            selected = [
                row for row in summary_rows
                if row["transform"] == transform
                and int(row["total_codebook_count"]) == int(count)
            ]
            if len(selected) == 2 and all(float(row["bootstrap_95_ci_high"]) <= 1.10 for row in selected):
                supported_candidates.append({"transform": transform, "count": int(count)})
    summary = {
        "schema_version": "wan5b_p1_canonical_summary.v1",
        "completed": p0.utc_now(),
        "config_hash": cfg_hash,
        "statistical_unit": "prompt; seeds and layer-heads averaged within prompt",
        "P1-C1_supported": bool(supported_candidates),
        "passing_candidates": supported_candidates,
        "decision_rule": "upper 95% prompt-bootstrap distortion/local ratio <=1.10 for both K and V at <=256 codebooks",
        "grouping_control": cfg["h5"]["grouping_policy"],
    }
    p0.atomic_json(root / "summary.json", summary)
    lines = [
        "# Wan2.2 5B P1 low-cost canonicalized sharing",
        "",
        f"Config hash: `{cfg_hash}`. Transform parameters use only each fold's 16 training prompts. P0 raw feature-cluster assignments are fixed across transforms.",
        "",
        "| Transform | Codebooks | Tensor | Distortion/local | 95% prompt CI | Improvement vs raw |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['transform']} | {row['total_codebook_count']} | {row['K_or_V']} | "
            f"{float(row['mean_distortion_to_local_ratio']):.4f} | "
            f"[{float(row['bootstrap_95_ci_low']):.4f}, {float(row['bootstrap_95_ci_high']):.4f}] | "
            f"{100*float(row['mean_improvement_vs_raw']):.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Preregistered P1-C1 verdict: **{'supported' if supported_candidates else 'not supported'}**.",
            "",
            "Rate accounting includes transform metadata stored in BF16. Dense rotations and transform-specific regrouping were intentionally excluded.",
        ]
    )
    report = root / "report.md"
    temp = report.with_suffix(f".md.tmp.{os.getpid()}")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp, report)
    p0.atomic_json(
        root / "summary.done.json",
        {"completed": p0.utc_now(), "config_hash": cfg_hash, "summary": "summary.json"},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "merge", "summarize"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--config", required=True)
    launch.add_argument("--gpus", default="")
    worker = sub.add_parser("worker")
    worker.add_argument("--config", required=True)
    worker.add_argument("--shard-id", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "launch":
        cmd_launch(args)
    elif args.command == "worker":
        cmd_worker(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
