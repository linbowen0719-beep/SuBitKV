#!/usr/bin/env python3
"""Prompt-held-out P1 study of compact residual RVQ schedules for Wan2.2 5B.

The P0 BF16 sample cache and P0 uniform-C64 fold codebooks are immutable
inputs.  Work is resumable at fold/seed/tensor/layer granularity, and every
launch writes to a distinct log directory so failed attempts remain auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


DIM = p0.DIM
LAYERS = p0.LAYERS
HEADS = p0.HEADS


def schedule_bits(schedule: Sequence[int]) -> int:
    bits = 0
    for clusters in schedule:
        clusters = int(clusters)
        if clusters < 2 or clusters & (clusters - 1):
            raise ValueError(f"Cluster count must be a power of two >=2, got {clusters}")
        bits += int(math.log2(clusters))
    return bits


def method_schedule(cfg: dict[str, Any], method: str) -> list[int]:
    if method.startswith("uniform_l") and method.endswith("_c64"):
        levels = int(method.removeprefix("uniform_l").removesuffix("_c64"))
        return [64] * levels
    return [int(value) for value in cfg["compact_schedules"][method]]


def rate_accounting(cfg: dict[str, Any], method: str) -> dict[str, Any]:
    schedule = method_schedule(cfg, method)
    logical_bits = schedule_bits(schedule)
    physical_bits = 8 * len(schedule)
    storage = str(cfg["rvq"]["runtime_codebook_storage"]).lower()
    entries = sum(schedule)
    if storage == "int4":
        # One signed INT4 value per coordinate plus one BF16 scale per centroid.
        codebook_bytes = entries * (DIM / 2 + 2)
    elif storage == "int8":
        codebook_bytes = entries * (DIM + 2)
    elif storage in {"bf16", "fp16"}:
        codebook_bytes = entries * DIM * 2
    else:
        codebook_bytes = entries * DIM * 4
    amortization = int(cfg["rvq"]["amortization_vectors_per_group"])
    logical_total = logical_bits + codebook_bytes * 8 / amortization
    physical_total = physical_bits + codebook_bytes * 8 / amortization
    groups_per_tensor = LAYERS * HEADS
    return {
        "method": method,
        "schedule": json.dumps(schedule, separators=(",", ":")),
        "levels": len(schedule),
        "centroid_count_per_group": entries,
        "logical_index_bits_per_vector": logical_bits,
        "current_uint8_index_bits_per_vector": physical_bits,
        "codebook_storage_bytes_per_group": codebook_bytes,
        "codebook_storage_mib_per_tensor_720_groups": codebook_bytes * groups_per_tensor / 1024**2,
        "codebook_storage_mib_K_plus_V_1440_groups": codebook_bytes * 2 * groups_per_tensor / 1024**2,
        "logical_total_amortized_bits_per_vector": logical_total,
        "current_uint8_total_amortized_bits_per_vector": physical_total,
        "amortization_vectors_per_group": amortization,
        "codebook_storage": storage,
    }


def p0_root(cfg: dict[str, Any]) -> Path:
    return p0.resolve_path(cfg["p0_result_dir"])


def output_root(cfg: dict[str, Any]) -> Path:
    return p0.resolve_path(cfg["output_dir"])


def axes(cfg: dict[str, Any]) -> tuple[list[int], list[int]]:
    return p0.parse_axis(cfg["layers"], LAYERS), p0.parse_axis(cfg["heads"], HEADS)


def fold_ids(cfg: dict[str, Any], fold: int) -> tuple[list[int], list[int]]:
    items = p0.prompt_items(Path(cfg["data_root"]))
    all_ids = [int(item["prompt_id"]) for item in items]
    folds = p0.make_folds(all_ids, int(cfg["folds"]), int(cfg["seed"]))
    test = list(folds[int(fold)])
    train = sorted(set(all_ids) - set(test))
    if len(train) != 16 or len(test) != 4:
        raise AssertionError(f"Expected prompt folds 16/4, got {len(train)}/{len(test)}")
    return train, test


def cached_layer(cfg: dict[str, Any], prompt_id: int, layer: int) -> dict[str, Any]:
    path = p0.cache_path(p0_root(cfg), int(prompt_id), int(layer))
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def collect_samples(
    cfg: dict[str, Any], tensor_name: str, layer: int, prompt_ids: Sequence[int], part: str
) -> torch.Tensor:
    _layers, head_ids = axes(cfg)
    train_take = int(cfg["sample_cache"]["train_samples_per_prompt_head"])
    eval_take = int(cfg["sample_cache"]["eval_samples_per_prompt_head"])
    parts = []
    for prompt_id in prompt_ids:
        tensor = cached_layer(cfg, int(prompt_id), int(layer))[tensor_name][head_ids]
        if part == "train":
            tensor = tensor[:, :train_take]
        elif part == "eval":
            tensor = tensor[:, -eval_take:]
        else:
            raise ValueError(part)
        parts.append(tensor)
    return torch.cat(parts, dim=1).contiguous()


def p0_codebook_path(
    cfg: dict[str, Any], fold: int, seed: int, tensor_name: str, layer: int
) -> Path:
    return (
        p0_root(cfg)
        / "cv"
        / f"fold_{fold}"
        / f"seed_{seed}"
        / tensor_name.lower()
        / "codebooks"
        / f"layer_{layer:02d}.pt"
    )


def load_uniform_codebooks(
    cfg: dict[str, Any], fold: int, seed: int, tensor_name: str, layer: int
) -> list[torch.Tensor]:
    _layers, head_ids = axes(cfg)
    path = p0_codebook_path(cfg, fold, seed, tensor_name, layer)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload["train_prompt_ids"] != fold_ids(cfg, fold)[0]:
        raise ValueError(f"P0 codebook prompt split mismatch: {path}")
    selected = payload["codebooks"][head_ids]
    return [selected[:, level].contiguous() for level in range(int(selected.shape[1]))]


def fit_variable_rvq_batched(
    x: torch.Tensor, schedule: Sequence[int], iterations: int, seed: int
) -> tuple[list[torch.Tensor], torch.Tensor]:
    residual = x.float().contiguous()
    codebooks: list[torch.Tensor] = []
    energies: list[torch.Tensor] = []
    for level, clusters in enumerate(schedule):
        centroids, ids, _counts = p0.kmeans_batched(
            residual, int(clusters), int(iterations), int(seed) + level * 1_000_003
        )
        residual = residual - p0.gather_centroids(centroids, ids)
        codebooks.append(centroids.detach().cpu().to(torch.bfloat16))
        energies.append((residual * residual).mean(dim=(1, 2)).detach().cpu())
    return codebooks, torch.stack(energies, dim=1)


def reconstruct(
    x: torch.Tensor, codebooks_cpu: Sequence[torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    reference = x.to(device=device, dtype=torch.float32)
    residual = reference.clone()
    reconstruction = torch.zeros_like(reference)
    all_ids: list[torch.Tensor] = []
    all_reconstruction: list[torch.Tensor] = []
    for codebook_cpu in codebooks_cpu:
        codebook = codebook_cpu.to(device=device, dtype=torch.float32)
        ids, _ratio = p0.assign_batched(residual, codebook)
        assigned = p0.gather_centroids(codebook, ids)
        reconstruction += assigned
        residual -= assigned
        all_ids.append(ids)
        all_reconstruction.append(reconstruction.clone())
    return reconstruction, all_ids, all_reconstruction


def unit_paths(
    root: Path, fold: int, seed: int, tensor_name: str, layer: int
) -> dict[str, Path]:
    base = root / "compact" / f"fold_{fold}" / f"seed_{seed}" / tensor_name.lower()
    return {
        "codebook": base / "codebooks" / f"layer_{layer:02d}.pt",
        "metrics": base / "metrics" / f"layer_{layer:02d}.csv",
        "usage": base / "usage" / f"layer_{layer:02d}.csv",
        "done": base / "done" / f"layer_{layer:02d}.json",
    }


def tasks(cfg: dict[str, Any]) -> list[tuple[int, int, str, int]]:
    layers, _heads = axes(cfg)
    return [
        (fold, int(seed), str(tensor_name), layer)
        for fold in range(int(cfg["folds"]))
        for seed in cfg["initialization_seeds"]
        for tensor_name in cfg["tensors"]
        for layer in layers
    ]


def method_codebooks(
    cfg: dict[str, Any], uniform: Sequence[torch.Tensor], compact: dict[str, Sequence[torch.Tensor]]
) -> dict[str, list[torch.Tensor]]:
    result = {
        f"uniform_l{int(levels)}_c64": list(uniform[: int(levels)])
        for levels in cfg["uniform_prefix_levels"]
    }
    result.update({name: list(value) for name, value in compact.items()})
    return result


def metric_rows_for_prompt(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    layer: int, prompt_id: int, methods: dict[str, list[torch.Tensor]], device: torch.device,
) -> list[dict[str, Any]]:
    _layers, heads = axes(cfg)
    x = collect_samples(cfg, tensor_name, layer, [prompt_id], "eval")
    rows: list[dict[str, Any]] = []
    for method, codebooks in methods.items():
        reconstruction, _ids, _stages = reconstruct(x, codebooks, device)
        values = p0.reconstruction_metrics(x.to(device=device).float(), reconstruction)
        rate = rate_accounting(cfg, method)
        for local_head, head in enumerate(heads):
            rows.append(
                {
                    "study_id": cfg["study_id"],
                    "config_hash": cfg_hash,
                    "hypothesis_id": "P1-R1,P1-R3",
                    "fold": fold,
                    "initialization_seed": seed,
                    "prompt_id": int(prompt_id),
                    "layer": layer,
                    "head": head,
                    "K_or_V": tensor_name,
                    "hook_stage": "pre_rope" if tensor_name == "K" else "pre_prope",
                    "method": method,
                    **{key: float(value[local_head].item()) for key, value in values.items()},
                    **{key: value for key, value in rate.items() if key != "method"},
                }
            )
    return rows


def usage_rows_for_split(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    layer: int, split: str, x: torch.Tensor, methods: dict[str, list[torch.Tensor]],
    device: torch.device,
) -> list[dict[str, Any]]:
    _layers, heads = axes(cfg)
    rows: list[dict[str, Any]] = []
    for method, codebooks in methods.items():
        _final, ids_by_stage, recon_by_stage = reconstruct(x, codebooks, device)
        reference = x.to(device=device, dtype=torch.float32)
        for level, (ids, reconstruction) in enumerate(zip(ids_by_stage, recon_by_stage), start=1):
            clusters = int(codebooks[level - 1].shape[1])
            usage = p0.usage_metrics(ids, clusters)
            quality = p0.reconstruction_metrics(reference, reconstruction)
            for local_head, head in enumerate(heads):
                rows.append(
                    {
                        "study_id": cfg["study_id"],
                        "config_hash": cfg_hash,
                        "hypothesis_id": "P1-R2",
                        "fold": fold,
                        "initialization_seed": seed,
                        "split": split,
                        "pooled_prompts": 16 if split == "train" else 4,
                        "samples_per_group": int(x.shape[1]),
                        "layer": layer,
                        "head": head,
                        "K_or_V": tensor_name,
                        "method": method,
                        "RVQ_stage": level,
                        "codebook_size": clusters,
                        "normalized_perplexity": float(
                            (usage["perplexity"][local_head] / clusters).item()
                        ),
                        **{
                            f"codebook_{key}": float(value[local_head].item())
                            for key, value in usage.items()
                        },
                        "relative_mse": float(quality["relative_mse"][local_head].item()),
                    }
                )
    return rows


def run_unit(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    layer: int, device: torch.device,
) -> None:
    paths = unit_paths(output_root(cfg), fold, seed, tensor_name, layer)
    if all(path.exists() for path in paths.values()):
        return
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite incomplete unit: {existing}")
    started = time.time()
    train_ids, test_ids = fold_ids(cfg, fold)
    train_cpu = collect_samples(cfg, tensor_name, layer, train_ids, "train")
    train = train_cpu.to(device=device, dtype=torch.float32)
    base_seed = (
        seed
        + fold * 10_000_019
        + layer * 1009
        + (0 if tensor_name == "K" else 500_009)
    )
    compact_codebooks: dict[str, list[torch.Tensor]] = {}
    train_energies: dict[str, torch.Tensor] = {}
    for name, schedule in cfg["compact_schedules"].items():
        codebooks, energies = fit_variable_rvq_batched(
            train,
            [int(value) for value in schedule],
            int(cfg["rvq"]["kmeans_iterations"]),
            base_seed,
        )
        compact_codebooks[str(name)] = codebooks
        train_energies[str(name)] = energies
    del train
    uniform = load_uniform_codebooks(cfg, fold, seed, tensor_name, layer)
    methods = method_codebooks(cfg, uniform, compact_codebooks)

    metric_rows: list[dict[str, Any]] = []
    for prompt_id in test_ids:
        metric_rows.extend(
            metric_rows_for_prompt(
                cfg, cfg_hash, fold, seed, tensor_name, layer, prompt_id, methods, device
            )
        )
    eval_cpu = collect_samples(cfg, tensor_name, layer, test_ids, "eval")
    usage_rows = usage_rows_for_split(
        cfg, cfg_hash, fold, seed, tensor_name, layer, "train", train_cpu, methods, device
    )
    usage_rows.extend(
        usage_rows_for_split(
            cfg, cfg_hash, fold, seed, tensor_name, layer, "heldout", eval_cpu, methods, device
        )
    )
    payload = {
        "schema_version": "wan5b_p1_variable_rvq.v1",
        "config_hash": cfg_hash,
        "fold": fold,
        "initialization_seed": seed,
        "tensor": tensor_name,
        "layer": layer,
        "heads": torch.tensor(axes(cfg)[1], dtype=torch.int16),
        "train_prompt_ids": train_ids,
        "test_prompt_ids": test_ids,
        "input_norm_mode": "none",
        "compact_schedules": {
            name: [int(value) for value in schedule]
            for name, schedule in cfg["compact_schedules"].items()
        },
        "compact_codebooks": compact_codebooks,
        "train_residual_mse": train_energies,
        "training_seconds": time.time() - started,
    }
    p0.atomic_torch(paths["codebook"], payload)
    p0.write_csv(paths["metrics"], metric_rows)
    p0.write_csv(paths["usage"], usage_rows)
    p0.atomic_json(
        paths["done"],
        {
            "completed": p0.utc_now(),
            "config_hash": cfg_hash,
            "fold": fold,
            "initialization_seed": seed,
            "tensor": tensor_name,
            "layer": layer,
            "metric_rows": len(metric_rows),
            "usage_rows": len(usage_rows),
            "codebook_sha256": p0.sha256_file(paths["codebook"]),
            "seconds": time.time() - started,
        },
    )
    print(
        f"[compact] fold={fold} seed={seed} tensor={tensor_name} layer={layer} "
        f"seconds={time.time()-started:.1f}",
        flush=True,
    )


def validate_inputs(cfg: dict[str, Any]) -> None:
    layers, heads = axes(cfg)
    if int(cfg["folds"]) != 5:
        raise ValueError("This preregistered study requires five 16/4 prompt folds")
    if not layers or not heads:
        raise ValueError("Empty layer/head axes")
    for name, schedule in cfg["compact_schedules"].items():
        bits = schedule_bits(schedule)
        required_bits = cfg.get("required_logical_index_bits")
        if required_bits is not None and bits != int(required_bits):
            raise ValueError(f"{name} has {bits} bits, expected {int(required_bits)}")
        if str(name).endswith("30") and bits != 30:
            raise ValueError(f"{name} has {bits} bits, expected 30")
        if str(name).endswith("36") and bits != 36:
            raise ValueError(f"{name} has {bits} bits, expected 36")
    parent_info_path = p0_root(cfg) / "run_info.json"
    parent_info = json.loads(parent_info_path.read_text(encoding="utf-8"))
    expected_folds = p0.make_folds(list(range(20)), int(cfg["folds"]), int(cfg["seed"]))
    if parent_info.get("folds") != expected_folds:
        raise ValueError("P0 result uses different prompt folds")
    expected_cache = 20 * LAYERS
    cache_files = list((p0_root(cfg) / "sample_cache").glob("prompt_*/layer_*.pt"))
    if len(cache_files) != expected_cache:
        raise ValueError(f"P0 sample cache incomplete: {len(cache_files)}/{expected_cache}")
    missing = [
        str(p0_codebook_path(cfg, fold, int(seed), tensor_name, layer))
        for fold in range(int(cfg["folds"]))
        for seed in cfg["initialization_seeds"]
        for tensor_name in cfg["tensors"]
        for layer in layers
        if not p0_codebook_path(cfg, fold, int(seed), tensor_name, layer).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} P0 codebook units; first={missing[:3]}")


def cmd_prepare(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    validate_inputs(cfg)
    root = output_root(cfg)
    run_info_path = root / "run_info.json"
    if root.exists() and not run_info_path.exists():
        raise FileExistsError(f"Refusing to adopt existing output without run_info.json: {root}")
    if run_info_path.exists():
        current = json.loads(run_info_path.read_text(encoding="utf-8"))
        if current.get("config_hash") != cfg_hash:
            raise ValueError("Existing output config hash differs")
        print(f"[prepare] already prepared root={root}", flush=True)
        return
    items = p0.prompt_items(Path(cfg["data_root"]))
    manifest_path = p0_root(cfg) / "cv" / "codebook_manifest.json"
    registry_path = p0.resolve_path(
        cfg.get("hypothesis_registry", "artifacts/p1_compact_hypotheses.csv")
    )
    folds = p0.make_folds(
        [int(item["prompt_id"]) for item in items], int(cfg["folds"]), int(cfg["seed"])
    )
    run_info = {
        "git_commit": p0.git_commit(),
        "config_path": str(cfg_path),
        "config_hash": cfg_hash,
        "codebook_path": {"p0_uniform_cv": str((p0_root(cfg) / "cv").resolve())},
        "codebook_hash": {"p0_uniform_cv_manifest": p0.sha256_file(manifest_path)},
        "hostname": p0.socket.gethostname(),
        "gpu_type": p0.gpu_name(),
        "seed": int(cfg["seed"]),
        "initialization_seeds": [int(value) for value in cfg["initialization_seeds"]],
        "prompt_list": [str(item["prompt"]) for item in items],
        "timestamp": p0.utc_now(),
        "model_id": cfg["model_id"],
        "source_sample_cache": str((p0_root(cfg) / "sample_cache").resolve()),
        "source_data_read_only": True,
        "statistical_unit": "prompt",
        "folds": folds,
        "hypothesis_registry": str(registry_path.resolve()),
        "hypothesis_registry_hash": p0.sha256_file(registry_path),
        "logical_rate_definition": "sum(log2(stage cardinality)); assumes packed indices",
        "physical_rate_caveat": "current implementation stores one uint8 index per RVQ stage",
    }
    p0.atomic_json(run_info_path, run_info)
    p0.atomic_json(root / "folds.json", {"config_hash": cfg_hash, "folds": folds})
    p0.write_csv(
        root / "tables" / "rate_accounting.csv",
        [
            rate_accounting(cfg, method)
            for method in [
                *[f"uniform_l{int(level)}_c64" for level in cfg["uniform_prefix_levels"]],
                *cfg["compact_schedules"].keys(),
            ]
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
        task for index, task in enumerate(tasks(cfg)) if index % args.num_shards == args.shard_id
    ]
    device = torch.device("cuda:0")
    print(
        f"[worker] shard={args.shard_id}/{args.num_shards} tasks={len(selected)} "
        f"gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    for task in selected:
        run_unit(cfg, cfg_hash, *task, device)


def cmd_launch(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, _cfg_hash = p0.parse_config(cfg_path)
    gpus = [int(value) for value in (args.gpus.split(",") if args.gpus else cfg["gpus"])]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError(f"P1 launch requires exactly four distinct GPUs, got {gpus}")
    launch_id = p0.utc_now().replace(":", "").replace("+", "_")
    log_dir = output_root(cfg) / "logs" / f"compact_{launch_id}"
    log_dir.mkdir(parents=True, exist_ok=False)
    processes = []
    for shard, gpu in enumerate(gpus):
        log_path = log_dir / f"gpu_{gpu}_shard_{shard}.log"
        handle = log_path.open("x", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            str(cfg["python"]),
            str(Path(__file__).resolve()),
            "worker",
            "--config",
            str(cfg_path),
            "--shard-id",
            str(shard),
            "--num-shards",
            str(len(gpus)),
        ]
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        processes.append((gpu, process, handle, log_path))
        print(f"[launch] gpu={gpu} pid={process.pid} log={log_path}", flush=True)
    last_report = 0.0
    try:
        while any(process.poll() is None for _gpu, process, _handle, _path in processes):
            if time.time() - last_report >= 10:
                status = ", ".join(
                    f"gpu{gpu}:pid{process.pid}:"
                    f"{'running' if process.poll() is None else process.returncode}"
                    for gpu, process, _handle, _path in processes
                )
                print(f"[monitor] {status}", flush=True)
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


def stream_csv(paths: Iterable[Path]) -> Iterable[dict[str, str]]:
    for path in sorted(paths):
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)


def cmd_merge(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    root = output_root(cfg)
    expected = len(tasks(cfg))
    done = list((root / "compact").glob("fold_*/seed_*/*/done/layer_*.json"))
    if len(done) != expected:
        raise RuntimeError(f"Incomplete compact run: {len(done)}/{expected} units")
    for path in done:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("config_hash") != cfg_hash:
            raise ValueError(f"Config hash mismatch in {path}")
    metric_rows = p0.merge_csv_glob(
        (root / "compact").glob("fold_*/seed_*/*/metrics/layer_*.csv"),
        root / "tables" / "compact_prompt_metrics.csv",
    )
    usage_rows = p0.merge_csv_glob(
        (root / "compact").glob("fold_*/seed_*/*/usage/layer_*.csv"),
        root / "tables" / "compact_usage.csv",
    )
    manifest_rows = []
    for path in sorted((root / "compact").glob("fold_*/seed_*/*/codebooks/layer_*.pt")):
        manifest_rows.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": p0.sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "wan5b_p1_compact_manifest.v1",
        "config_hash": cfg_hash,
        "files": manifest_rows,
        "total_bytes": sum(int(row["bytes"]) for row in manifest_rows),
    }
    manifest_path = root / "compact" / "codebook_manifest.json"
    p0.atomic_json(manifest_path, manifest)
    info_path = root / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["output_codebook_path"] = str((root / "compact").resolve())
    info["output_codebook_hash"] = p0.sha256_file(manifest_path)
    p0.atomic_json(info_path, info)
    p0.atomic_json(
        root / "compact" / "compact.done.json",
        {
            "completed": p0.utc_now(),
            "config_hash": cfg_hash,
            "units": expected,
            "prompt_metric_rows": metric_rows,
            "usage_rows": usage_rows,
            "codebook_files": len(manifest_rows),
            "codebook_bytes": manifest["total_bytes"],
        },
    )
    print(
        f"[merge] units={expected} prompt_rows={metric_rows} usage_rows={usage_rows}",
        flush=True,
    )


def bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float("nan"), float("nan")
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    batch = 1000
    for start in range(0, int(samples), batch):
        end = min(int(samples), start + batch)
        ids = generator.integers(0, array.size, size=(end - start, array.size))
        means[start:end] = array[ids].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_prompt_metrics(cfg: dict[str, Any], root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sums: dict[tuple[str, str, int], list[float]] = {}
    metric_path = root / "tables" / "compact_prompt_metrics.csv"
    for row in stream_csv([metric_path]):
        key = (row["method"], row["K_or_V"], int(row["prompt_id"]))
        aggregate = sums.setdefault(key, [0.0, 0.0])
        aggregate[0] += float(row["relative_mse"])
        aggregate[1] += 1
    prompt_rows = [
        {
            "method": method,
            "K_or_V": tensor_name,
            "prompt_id": prompt_id,
            "relative_mse": total / count,
            "group_seed_rows": int(count),
        }
        for (method, tensor_name, prompt_id), (total, count) in sorted(sums.items())
    ]
    lookup = {
        (str(row["method"]), str(row["K_or_V"]), int(row["prompt_id"])): float(row["relative_mse"])
        for row in prompt_rows
    }
    methods = sorted({str(row["method"]) for row in prompt_rows})
    prompts = sorted({int(row["prompt_id"]) for row in prompt_rows})
    for method in methods:
        for prompt_id in prompts:
            values = [lookup[(method, tensor_name, prompt_id)] for tensor_name in ("K", "V")]
            lookup[(method, "K+V", prompt_id)] = sum(values) / len(values)
            prompt_rows.append(
                {
                    "method": method,
                    "K_or_V": "K+V",
                    "prompt_id": prompt_id,
                    "relative_mse": lookup[(method, "K+V", prompt_id)],
                    "group_seed_rows": sum(
                        int(next(row["group_seed_rows"] for row in prompt_rows if row["method"] == method and row["K_or_V"] == tensor_name and row["prompt_id"] == prompt_id))
                        for tensor_name in ("K", "V")
                    ),
                }
            )
    comparisons = [("compact30", "uniform_l5_c64"), ("compact36", "uniform_l6_c64")]
    paired_rows: list[dict[str, Any]] = []
    bootstrap_samples = int(cfg["bootstrap"]["samples"])
    bootstrap_seed = int(cfg["bootstrap"]["seed"])
    for comparison_index, (candidate, baseline) in enumerate(comparisons):
        for tensor_index, tensor_name in enumerate(("K", "V", "K+V")):
            improvements = []
            deltas = []
            for prompt_id in prompts:
                base = lookup[(baseline, tensor_name, prompt_id)]
                cand = lookup[(candidate, tensor_name, prompt_id)]
                improvements.append((base - cand) / max(base, 1e-20))
                deltas.append(cand - base)
            ci_low, ci_high = bootstrap_ci(
                improvements,
                bootstrap_samples,
                bootstrap_seed + comparison_index * 101 + tensor_index,
            )
            paired_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "K_or_V": tensor_name,
                    "prompts": len(improvements),
                    "mean_relative_mse_improvement": float(np.mean(improvements)),
                    "median_relative_mse_improvement": float(np.median(improvements)),
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                    "prompt_win_rate": float(np.mean(np.asarray(improvements) > 0)),
                    "mean_absolute_relative_mse_delta": float(np.mean(deltas)),
                }
            )
    return prompt_rows, paired_rows


def summarize_usage(root: Path) -> list[dict[str, Any]]:
    values: dict[tuple[str, str, str, int], dict[str, list[float]]] = {}
    for row in stream_csv([root / "tables" / "compact_usage.csv"]):
        key = (row["method"], row["K_or_V"], row["split"], int(row["RVQ_stage"]))
        metrics = values.setdefault(
            key,
            {
                "normalized_perplexity": [],
                "dead_code_ratio": [],
                "top_code_frequency": [],
                "relative_mse": [],
            },
        )
        metrics["normalized_perplexity"].append(float(row["normalized_perplexity"]))
        metrics["dead_code_ratio"].append(float(row["codebook_dead_code_ratio"]))
        metrics["top_code_frequency"].append(float(row["codebook_top_code_frequency"]))
        metrics["relative_mse"].append(float(row["relative_mse"]))
    result = []
    for (method, tensor_name, split, stage), metrics in sorted(values.items()):
        row: dict[str, Any] = {
            "method": method,
            "K_or_V": tensor_name,
            "split": split,
            "RVQ_stage": stage,
            "fold_seed_layer_head_units": len(metrics["normalized_perplexity"]),
        }
        for metric, metric_values in metrics.items():
            array = np.asarray(metric_values)
            row[f"mean_{metric}"] = float(array.mean())
            row[f"median_{metric}"] = float(np.median(array))
            row[f"p10_{metric}"] = float(np.quantile(array, 0.1))
            row[f"p90_{metric}"] = float(np.quantile(array, 0.9))
        result.append(row)
    return result


def cmd_summarize(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    root = output_root(cfg)
    done_path = root / "compact" / "compact.done.json"
    if not done_path.exists():
        raise FileNotFoundError(f"Run merge first: {done_path}")
    prompt_rows, paired_rows = summarize_prompt_metrics(cfg, root)
    usage_rows = summarize_usage(root)
    p0.write_csv(root / "tables" / "prompt_aggregate.csv", prompt_rows)
    p0.write_csv(root / "tables" / "paired_compact_comparisons.csv", paired_rows)
    p0.write_csv(root / "tables" / "usage_summary.csv", usage_rows)
    paired = {(row["candidate"], row["K_or_V"]): row for row in paired_rows}
    verdicts = {}
    for candidate in ("compact30", "compact36"):
        joint = paired[(candidate, "K+V")]
        no_tensor_harm = all(
            float(paired[(candidate, tensor_name)]["bootstrap_95_ci_low"]) >= -0.02
            for tensor_name in ("K", "V")
        )
        verdicts[candidate] = {
            "P1-R1_supported": float(joint["bootstrap_95_ci_low"]) > 0 and no_tensor_harm,
            "joint_improvement": joint,
            "no_tensor_upper_95pct_degradation_above_2pct": no_tensor_harm,
        }
    late_usage = [
        row
        for row in usage_rows
        if row["method"] == "uniform_l8_c64"
        and row["split"] == "heldout"
        and int(row["RVQ_stage"]) in (7, 8)
    ]
    usage_by_stage = {
        stage: float(
            np.median(
                [
                    float(row["median_normalized_perplexity"])
                    for row in late_usage
                    if int(row["RVQ_stage"]) == stage
                ]
            )
        )
        for stage in (7, 8)
    }
    r2_supported = all(value < 0.40 for value in usage_by_stage.values())
    rates = {
        method: rate_accounting(cfg, method)
        for method in ("uniform_l5_c64", "uniform_l6_c64", "compact30", "compact36")
    }
    r3 = {
        "compact30_codebook_reduction_vs_uniform_l5": 1
        - rates["compact30"]["codebook_storage_bytes_per_group"]
        / rates["uniform_l5_c64"]["codebook_storage_bytes_per_group"],
        "compact36_codebook_reduction_vs_uniform_l6": 1
        - rates["compact36"]["codebook_storage_bytes_per_group"]
        / rates["uniform_l6_c64"]["codebook_storage_bytes_per_group"],
        "current_uint8_physical_index_caveat": (
            "compact schedules use eight uint8 stage tensors (64 bits/vector) until variable-bit packing exists"
        ),
    }
    summary = {
        "schema_version": "wan5b_p1_compact_summary.v1",
        "completed": p0.utc_now(),
        "config_hash": cfg_hash,
        "statistical_unit": "prompt; initialization seeds and layer-heads averaged within prompt",
        "P1-R1": verdicts,
        "P1-R2": {
            "supported": r2_supported,
            "uniform_l8_heldout_median_normalized_perplexity": usage_by_stage,
            "threshold": 0.40,
        },
        "P1-R3": {"supported": True, **r3},
    }
    p0.atomic_json(root / "summary.json", summary)
    lines = [
        "# Wan2.2 5B P1 compact residual RVQ",
        "",
        f"Config hash: `{cfg_hash}`. Primary statistical unit: prompt; seeds and layer-heads are averaged within each prompt.",
        "",
        "## P1-R1: equal logical-index reconstruction",
        "",
        "| Candidate | Tensor | Mean improvement | 95% prompt bootstrap CI | Win rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            f"| {row['candidate']} vs {row['baseline']} | {row['K_or_V']} | "
            f"{100*float(row['mean_relative_mse_improvement']):.3f}% | "
            f"[{100*float(row['bootstrap_95_ci_low']):.3f}%, "
            f"{100*float(row['bootstrap_95_ci_high']):.3f}%] | "
            f"{100*float(row['prompt_win_rate']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## P1-R2: corrected late-stage usage",
            "",
            f"Uniform L8C64 held-out median normalized perplexity: stage 7 = {usage_by_stage[7]:.4f}, stage 8 = {usage_by_stage[8]:.4f}. Preregistered threshold: <0.40 at both stages.",
            "",
            "## P1-R3: storage accounting",
            "",
            f"Compact30 removes {100*r3['compact30_codebook_reduction_vs_uniform_l5']:.2f}% of centroid storage versus L5C64; compact36 removes {100*r3['compact36_codebook_reduction_vs_uniform_l6']:.2f}% versus L6C64.",
            "",
            "The logical matched-rate claim assumes bit-packed variable-cardinality indices. The current runtime uses one uint8 tensor per stage, so both eight-stage compact schedules physically consume 64 index bits/vector until a packing kernel is implemented.",
            "",
            "Raw prompt rows, fold-level usage diagnostics, and deterministic accounting are in `tables/`.",
        ]
    )
    report_path = root / "report.md"
    tmp = report_path.with_suffix(f".md.tmp.{os.getpid()}")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, report_path)
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
