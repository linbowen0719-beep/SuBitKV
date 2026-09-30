#!/usr/bin/env python3
"""Hypothesis-driven P0 analysis for Wan2.2 5B historical self-attention KV.

The raw KV dump is always read-only.  Expensive work is split into resumable
prompt/layer sampling units and fold/seed/tensor/layer codebook units so a
failed worker can be restarted without overwriting completed evidence.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENS_PER_FRAME = 880
SOURCE_CHUNK_FRAMES = 4
LAYERS = 30
HEADS = 24
DIM = 128


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_config(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return cfg, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_axis(value: Any, maximum: int) -> list[int]:
    if isinstance(value, str) and value.lower() == "all":
        return list(range(maximum))
    result = [int(item) for item in value]
    if not result or min(result) < 0 or max(result) >= maximum:
        raise ValueError(f"Invalid axis values {result}; expected values in [0,{maximum})")
    return sorted(set(result))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*.pt") if item.is_file())
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() or "unknown"


def gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def prompt_items(data_root: Path) -> list[dict[str, Any]]:
    with (data_root / "prompt_split.json").open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    items = list(split["train"])
    if len(items) != 20:
        raise ValueError(f"Expected exactly 20 calibration prompts, found {len(items)}")
    return sorted(items, key=lambda item: int(item["prompt_id"]))


def make_folds(prompt_ids: Sequence[int], folds: int, seed: int) -> list[list[int]]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.tensor(list(prompt_ids), dtype=torch.long)
    order = order[torch.randperm(order.numel(), generator=generator)].tolist()
    result = [[] for _ in range(folds)]
    for index, prompt_id in enumerate(order):
        result[index % folds].append(int(prompt_id))
    if sorted(x for fold in result for x in fold) != sorted(prompt_ids):
        raise AssertionError("Fold construction lost or duplicated prompts")
    return [sorted(fold) for fold in result]


def output_root(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["output_dir"])


def cache_path(root: Path, prompt_id: int, layer: int) -> Path:
    return root / "sample_cache" / f"prompt_{prompt_id:02d}" / f"layer_{layer:02d}.pt"


def audit_partial_path(root: Path, prompt_id: int, layer: int) -> Path:
    return root / "audit" / "partials" / f"prompt_{prompt_id:02d}_layer_{layer:02d}.json"


def raw_layer_path(data_root: Path, prompt_id: int, layer: int) -> Path:
    return data_root / "kv_dump" / "train" / str(prompt_id) / f"layer_{layer}.pt"


def stratified_indices(
    *, total_frames: int, head: int, samples: int, seed: int, prompt_id: int, layer: int
) -> torch.Tensor:
    chunk_count = math.ceil(total_frames / SOURCE_CHUNK_FRAMES)
    per_chunk = math.ceil(samples / chunk_count)
    chunk_parts: list[torch.Tensor] = []
    for chunk in range(chunk_count):
        frame_start = chunk * SOURCE_CHUNK_FRAMES
        frame_end = min(total_frames, frame_start + SOURCE_CHUNK_FRAMES)
        count = (frame_end - frame_start) * TOKENS_PER_FRAME
        take = min(per_chunk, count)
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + prompt_id * 10_000_019 + layer * 100_003 + head * 1009 + chunk * 97
        )
        local = torch.randperm(count, generator=generator)[:take]
        chunk_parts.append(local + frame_start * TOKENS_PER_FRAME)
    width = max(part.numel() for part in chunk_parts)
    interleaved: list[int] = []
    for offset in range(width):
        for part in chunk_parts:
            if offset < part.numel():
                interleaved.append(int(part[offset]))
    return torch.tensor(interleaved[:samples], dtype=torch.long)


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    if contiguous.dtype == torch.bfloat16:
        contiguous = contiguous.view(torch.uint16)
    return contiguous.numpy().tobytes(order="C")


def vector_hashes(samples: torch.Tensor) -> Counter[str]:
    counts: Counter[str] = Counter()
    flat = samples.reshape(-1, samples.shape[-1])
    for vector in flat:
        counts[hashlib.sha256(tensor_bytes(vector)).hexdigest()] += 1
    return counts


def prepare_unit(
    cfg: dict[str, Any], cfg_hash: str, item: dict[str, Any], layer: int
) -> tuple[Path, Path]:
    root = output_root(cfg)
    data_root = Path(cfg["data_root"])
    prompt_id = int(item["prompt_id"])
    out_cache = cache_path(root, prompt_id, layer)
    out_audit = audit_partial_path(root, prompt_id, layer)
    if out_cache.exists() and out_audit.exists():
        return out_cache, out_audit

    heads = parse_axis(cfg["heads"], HEADS)
    sample_cfg = cfg["sample_cache"]
    sample_count = int(sample_cfg["samples_per_prompt_head"])
    path = raw_layer_path(data_root, prompt_id, layer)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    metadata = dict(payload.get("metadata", {}))
    for tensor_name in ("Q", "K", "V"):
        tensor = payload[tensor_name]
        if tensor.ndim != 4 or tensor.shape[1:] != (HEADS, TOKENS_PER_FRAME, DIM):
            raise ValueError(f"Unexpected {tensor_name} shape in {path}: {tuple(tensor.shape)}")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 {tensor_name} in {path}, got {tensor.dtype}")

    total_frames = int(payload["K"].shape[0])
    indices = torch.stack(
        [
            stratified_indices(
                total_frames=total_frames,
                head=head,
                samples=sample_count,
                seed=int(cfg["seed"]),
                prompt_id=prompt_id,
                layer=layer,
            )
            for head in heads
        ],
        dim=0,
    )
    sample_payload: dict[str, Any] = {
        "schema_version": "wan5b_rvq_sample_cache.v1",
        "config_hash": cfg_hash,
        "prompt_id": prompt_id,
        "layer": int(layer),
        "heads": torch.tensor(heads, dtype=torch.int16),
        "flat_token_indices": indices,
        "source_chunks": (indices // (SOURCE_CHUNK_FRAMES * TOKENS_PER_FRAME)).to(torch.int16),
        "metadata": metadata,
    }
    for tensor_name in ("Q", "K", "V"):
        source = payload[tensor_name][:, heads].permute(1, 0, 2, 3).reshape(len(heads), -1, DIM)
        sample_payload[tensor_name] = source.gather(
            1, indices[:, :, None].expand(-1, -1, DIM)
        ).contiguous()
    atomic_torch(out_cache, sample_payload)

    hash_all = bool(sample_cfg.get("hash_all_source_chunks", False))
    chunk_rows: list[dict[str, Any]] = []
    for tensor_name in ("K", "V"):
        tensor = payload[tensor_name]
        for source_chunk, frame_start in enumerate(range(0, total_frames, SOURCE_CHUNK_FRAMES)):
            frame_end = min(total_frames, frame_start + SOURCE_CHUNK_FRAMES)
            chunk_tensor = tensor[frame_start:frame_end]
            if hash_all:
                digest = hashlib.sha256(tensor_bytes(chunk_tensor)).hexdigest()
                hash_scope = "full_source_chunk_all_heads"
                byte_count = int(chunk_tensor.numel() * chunk_tensor.element_size())
            else:
                selected = sample_payload[tensor_name][
                    sample_payload["source_chunks"] == source_chunk
                ]
                digest = hashlib.sha256(tensor_bytes(selected)).hexdigest()
                hash_scope = "smoke_sample_only"
                byte_count = int(selected.numel() * selected.element_size())
            chunk_rows.append(
                {
                    "prompt_id": prompt_id,
                    "generation_id": f"prompt_{prompt_id}_seed_{cfg['seed']}",
                    "layer": int(layer),
                    "K_or_V": tensor_name,
                    "hook_stage": "pre_rope" if tensor_name == "K" else "pre_prope",
                    "source_chunk": source_chunk,
                    "read_chunk": None,
                    "history_age": None,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "sha256": digest,
                    "hash_scope": hash_scope,
                    "bytes": byte_count,
                }
            )
    vector_summary = {}
    for tensor_name in ("K", "V"):
        counts = vector_hashes(sample_payload[tensor_name])
        vector_summary[tensor_name] = {
            "sample_vectors": int(sum(counts.values())),
            "unique_vectors": int(len(counts)),
            "duplicate_vectors": int(sum(value - 1 for value in counts.values())),
            "duplicate_hashes": {key: value for key, value in counts.items() if value > 1},
        }
    audit = {
        "schema_version": "wan5b_kv_canonical_audit.v1",
        "config_hash": cfg_hash,
        "prompt_id": prompt_id,
        "layer": int(layer),
        "raw_path": str(path),
        "raw_size_bytes": path.stat().st_size,
        "metadata": metadata,
        "tensor_shapes": {name: list(payload[name].shape) for name in ("Q", "K", "V")},
        "tensor_dtypes": {name: str(payload[name].dtype) for name in ("Q", "K", "V")},
        "chunk_rows": chunk_rows,
        "vector_hash_summary": vector_summary,
        "snapshot_semantics": "canonical_concatenation_of_unique_append_spans",
    }
    atomic_json(out_audit, audit)
    return out_cache, out_audit


def write_run_info(cfg: dict[str, Any], cfg_path: Path, cfg_hash: str) -> None:
    root = output_root(cfg)
    items = prompt_items(Path(cfg["data_root"]))
    h6 = cfg.get("h6", {})
    cb_root = Path(h6.get("codebook_root", cfg["data_root"]))
    codebook_paths = {
        "k": str(cb_root / h6.get("k_codebook_dir", "")),
        "v": str(cb_root / h6.get("v_codebook_dir", "")),
    }
    codebook_hashes = {}
    for key, value in codebook_paths.items():
        path = Path(value)
        codebook_hashes[key] = tree_hash(path) if path.name else None
    payload = {
        "git_commit": git_commit(),
        "config_path": str(cfg_path.resolve()),
        "config_hash": cfg_hash,
        "codebook_path": codebook_paths,
        "codebook_hash": codebook_hashes,
        "hostname": socket.gethostname(),
        "gpu_type": gpu_name(),
        "seed": int(cfg["seed"]),
        "prompt_list": [item["prompt"] for item in items],
        "timestamp": utc_now(),
        "model_id": cfg["model_id"],
        "source_data": str(Path(cfg["data_root"]) / "kv_dump" / "train"),
        "source_data_read_only": True,
        "statistical_unit": "prompt",
        "folds": make_folds([int(item["prompt_id"]) for item in items], int(cfg["folds"]), int(cfg["seed"])),
        "initialization_seeds": list(cfg["initialization_seeds"]),
        "platform": platform.platform(),
    }
    atomic_json(root / "run_info.json", payload)


def aggregate_audit(cfg: dict[str, Any], cfg_hash: str) -> None:
    root = output_root(cfg)
    partials = sorted((root / "audit" / "partials").glob("*.json"))
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in partials]
    chunk_rows = [row for audit in audits for row in audit["chunk_rows"]]
    write_csv(root / "audit" / "source_chunk_hashes.csv", chunk_rows)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in chunk_rows:
        grouped[(str(row["K_or_V"]), int(row["layer"]))].append(row)
    duplicate_rows = []
    for (tensor_name, layer), rows in sorted(grouped.items()):
        counts = Counter(str(row["sha256"]) for row in rows)
        duplicate_rows.append(
            {
                "K_or_V": tensor_name,
                "layer": layer,
                "chunks": len(rows),
                "unique_chunk_hashes": len(counts),
                "duplicate_chunk_instances": sum(value - 1 for value in counts.values()),
                "duplicate_ratio": sum(value - 1 for value in counts.values()) / max(1, len(rows)),
                "hash_scope": rows[0]["hash_scope"],
            }
        )
    write_csv(root / "audit" / "duplicate_summary.csv", duplicate_rows)

    vector_rows = []
    for audit in audits:
        for tensor_name, summary in audit["vector_hash_summary"].items():
            vector_rows.append(
                {
                    "prompt_id": audit["prompt_id"],
                    "layer": audit["layer"],
                    "K_or_V": tensor_name,
                    **{key: value for key, value in summary.items() if key != "duplicate_hashes"},
                }
            )
    write_csv(root / "audit" / "sample_vector_hash_summary.csv", vector_rows)

    spans_root = Path(cfg["data_root"]) / "kv_dump" / "train"
    span_files = list(spans_root.glob("*/_spans/**/*.pt"))
    implementation_audit = {
        "schema_version": "wan5b_cache_implementation_audit.v1",
        "config_hash": cfg_hash,
        "raw_files_audited": len(audits),
        "expected_files": len(prompt_items(Path(cfg["data_root"]))) * len(parse_axis(cfg["layers"], LAYERS)),
        "span_files_remaining_after_merge": len(span_files),
        "merged_file_semantics": "Each THND file concatenates disjoint append spans along latent time exactly once.",
        "cumulative_snapshot_saved": False,
        "read_chunk_available": False,
        "canonicalization_action": "No vector-level deduplication needed; source chunks are already canonical append spans.",
        "code_evidence": [
            "scripts/rvq/diag/run_rvq_diag.py:693-785",
            "experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py:739-777",
            "subitkv/kv_cache.py:19-150",
        ],
        "data_evidence": [str(root / "audit" / "source_chunk_hashes.csv"), str(root / "audit" / "duplicate_summary.csv")],
    }
    atomic_json(root / "audit" / "implementation_audit.json", implementation_audit)
    atomic_json(root / "audit" / "prepare.done.json", {"completed": utc_now(), "config_hash": cfg_hash})


def cmd_prepare(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    root = output_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    items = prompt_items(Path(cfg["data_root"]))
    layers = parse_axis(cfg["layers"], LAYERS)
    folds = make_folds([int(item["prompt_id"]) for item in items], int(cfg["folds"]), int(cfg["seed"]))
    atomic_json(root / "folds.json", {"config_hash": cfg_hash, "folds": folds})
    write_run_info(cfg, cfg_path, cfg_hash)
    total = len(items) * len(layers)
    completed = 0
    started = time.time()
    for item in items:
        for layer in layers:
            prepare_unit(cfg, cfg_hash, item, layer)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == total:
                elapsed = time.time() - started
                print(f"[prepare] {completed}/{total} units elapsed={elapsed:.1f}s", flush=True)
    aggregate_audit(cfg, cfg_hash)
    print(f"[prepare] complete root={root} config_hash={cfg_hash}", flush=True)


def load_cached(cfg: dict[str, Any], prompt_id: int, layer: int) -> dict[str, Any]:
    path = cache_path(output_root(cfg), prompt_id, layer)
    if not path.exists():
        raise FileNotFoundError(f"Missing sample cache {path}; run prepare first")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def fold_ids(cfg: dict[str, Any], fold: int) -> tuple[list[int], list[int]]:
    items = prompt_items(Path(cfg["data_root"]))
    folds = make_folds([int(item["prompt_id"]) for item in items], int(cfg["folds"]), int(cfg["seed"]))
    test = list(folds[fold])
    train = sorted(set(int(item["prompt_id"]) for item in items) - set(test))
    if len(train) != 16 or len(test) != 4:
        raise AssertionError(f"Expected 16/4 fold, got {len(train)}/{len(test)}")
    return train, test


def collect_samples(
    cfg: dict[str, Any], tensor_name: str, layer: int, prompt_ids: Sequence[int], *, part: str
) -> torch.Tensor:
    train_take = int(cfg["sample_cache"]["train_samples_per_prompt_head"])
    eval_take = int(cfg["sample_cache"]["eval_samples_per_prompt_head"])
    parts = []
    for prompt_id in prompt_ids:
        payload = load_cached(cfg, int(prompt_id), layer)
        tensor = payload[tensor_name]
        if part == "train":
            selected = tensor[:, :train_take]
        elif part == "eval":
            selected = tensor[:, -eval_take:]
        elif part == "oracle_train":
            selected = tensor[:, :-eval_take]
        else:
            raise ValueError(part)
        parts.append(selected)
    return torch.cat(parts, dim=1).contiguous()


def assign_batched(
    x: torch.Tensor, centroids: torch.Tensor, *, return_ratio: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x32 = x.float()
    c32 = centroids.float()
    distances = (
        (x32 * x32).sum(dim=-1, keepdim=True)
        + (c32 * c32).sum(dim=-1)[:, None, :]
        - 2.0 * torch.bmm(x32, c32.transpose(1, 2))
    ).clamp_min_(0.0)
    if return_ratio:
        nearest = torch.topk(distances, k=2, dim=-1, largest=False)
        ratio = nearest.values[..., 0] / nearest.values[..., 1].clamp_min(1e-12)
        return nearest.indices[..., 0], ratio
    return torch.argmin(distances, dim=-1), None


def gather_centroids(centroids: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return centroids.gather(1, ids[..., None].expand(-1, -1, centroids.shape[-1]))


def kmeans_batched(
    x: torch.Tensor, clusters: int, iterations: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, count, dim = x.shape
    if count < clusters:
        raise ValueError(f"KMeans needs at least {clusters} samples, got {count}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    init_cpu = torch.stack(
        [torch.randperm(count, generator=generator)[:clusters] for _ in range(batch)], dim=0
    )
    centroids = x.gather(1, init_cpu.to(x.device)[..., None].expand(-1, -1, dim)).float().clone()
    ids = torch.zeros((batch, count), dtype=torch.long, device=x.device)
    counts = torch.zeros((batch, clusters), dtype=torch.float32, device=x.device)
    for _ in range(max(1, iterations)):
        ids, _ = assign_batched(x, centroids)
        sums = torch.zeros_like(centroids)
        counts = torch.zeros((batch, clusters), dtype=torch.float32, device=x.device)
        sums.scatter_add_(1, ids[..., None].expand(-1, -1, dim), x.float())
        counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float32))
        updated = sums / counts[..., None].clamp_min(1.0)
        empty = counts <= 0
        centroids = torch.where(empty[..., None], centroids, updated)
    ids, _ = assign_batched(x, centroids)
    return centroids, ids, counts


def fit_rvq_batched(
    x: torch.Tensor, *, levels: int, clusters: int, iterations: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = x.float().contiguous()
    codebooks = []
    residual_energy = []
    for level in range(levels):
        centroids, ids, _ = kmeans_batched(
            residual, clusters, iterations, seed + level * 1_000_003
        )
        residual = residual - gather_centroids(centroids, ids)
        codebooks.append(centroids.detach().cpu().to(torch.bfloat16))
        residual_energy.append((residual * residual).mean(dim=(1, 2)).detach().cpu())
    return torch.stack(codebooks, dim=1), torch.stack(residual_energy, dim=1)


def random_rvq_codebooks(
    x: torch.Tensor, *, levels: int, clusters: int, seed: int
) -> torch.Tensor:
    residual = x.float().contiguous()
    result = []
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 9187)
    for _level in range(levels):
        indices = torch.stack(
            [torch.randperm(residual.shape[1], generator=generator)[:clusters] for _ in range(residual.shape[0])]
        ).to(residual.device)
        centroids = residual.gather(1, indices[..., None].expand(-1, -1, residual.shape[-1]))
        ids, _ = assign_batched(residual, centroids)
        residual = residual - gather_centroids(centroids, ids)
        result.append(centroids.detach().cpu().to(torch.bfloat16))
    return torch.stack(result, dim=1)


def effective_ranks(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = residual.float() - residual.float().mean(dim=1, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular * singular
    probs = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy_rank = torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1))
    stable_rank = energy.sum(dim=-1) / energy.max(dim=-1).values.clamp_min(1e-12)
    return entropy_rank, stable_rank


def usage_metrics(ids: torch.Tensor, clusters: int) -> dict[str, torch.Tensor]:
    counts = torch.zeros((ids.shape[0], clusters), dtype=torch.float32, device=ids.device)
    counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float32))
    probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=-1)
    return {
        "entropy_bits": entropy,
        "perplexity": torch.pow(2.0, entropy),
        "dead_code_ratio": (counts == 0).float().mean(dim=-1),
        "top_code_frequency": probs.max(dim=-1).values,
    }


def reconstruction_metrics(x: torch.Tensor, recon: torch.Tensor) -> dict[str, torch.Tensor]:
    x32 = x.float()
    r32 = recon.float()
    error = r32 - x32
    vector_mse = (error * error).mean(dim=-1)
    vector_energy = (x32 * x32).mean(dim=-1)
    mse = vector_mse.mean(dim=-1)
    rel = vector_mse.sum(dim=-1) / vector_energy.sum(dim=-1).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(x32, r32, dim=-1).mean(dim=-1)
    norm_dist = (
        torch.linalg.vector_norm(r32, dim=-1) - torch.linalg.vector_norm(x32, dim=-1)
    ).abs() / torch.linalg.vector_norm(x32, dim=-1).clamp_min(1e-8)
    return {
        "mse": mse,
        "relative_mse": rel,
        "normalized_rmse": torch.sqrt(rel),
        "cosine_similarity": cosine,
        "p95_vector_mse": torch.quantile(vector_mse, 0.95, dim=-1),
        "p99_vector_mse": torch.quantile(vector_mse, 0.99, dim=-1),
        "max_vector_mse": vector_mse.max(dim=-1).values,
        "vector_norm_distortion": norm_dist.mean(dim=-1),
    }


def codebook_rate(cfg: dict[str, Any], levels: int) -> dict[str, float]:
    clusters = int(cfg["rvq"]["codebook_size"])
    logical_index_bits = levels * math.ceil(math.log2(clusters))
    storage = str(cfg["rvq"]["runtime_codebook_storage"]).lower()
    values = levels * clusters * DIM
    centroid_vectors = levels * clusters
    if storage == "int4":
        cb_bytes = math.ceil(values / 2) + centroid_vectors * 2
    elif storage == "int8":
        cb_bytes = values + centroid_vectors * 2
    elif storage in {"bf16", "fp16"}:
        cb_bytes = values * 2
    else:
        cb_bytes = values * 4
    amort = int(cfg["rvq"]["amortization_vectors_per_group"])
    total_bits = logical_index_bits + cb_bytes * 8 / max(1, amort)
    return {
        "logical_index_bits_per_vector": float(logical_index_bits),
        "index_bits_per_scalar": logical_index_bits / DIM,
        "codebook_storage_bytes_per_group": float(cb_bytes),
        "total_amortized_bits_per_vector": float(total_bits),
        "total_amortized_bits_per_scalar": total_bits / DIM,
    }


def cv_task_paths(root: Path, fold: int, seed: int, tensor_name: str, layer: int) -> dict[str, Path]:
    base = root / "cv" / f"fold_{fold}" / f"seed_{seed}" / tensor_name.lower()
    return {
        "codebook": base / "codebooks" / f"layer_{layer:02d}.pt",
        "metrics": base / "partials" / f"layer_{layer:02d}_prompt_metrics.csv",
        "stage": base / "partials" / f"layer_{layer:02d}_stagewise.csv",
        "done": base / "done" / f"layer_{layer:02d}.json",
    }


def evaluate_codebooks(
    cfg: dict[str, Any], fold: int, seed: int, tensor_name: str, layer: int,
    codebooks_cpu: torch.Tensor, random_codebooks_cpu: torch.Tensor, test_ids: Sequence[int], device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    heads = parse_axis(cfg["heads"], HEADS)
    levels = int(cfg["rvq"]["levels"])
    clusters = int(cfg["rvq"]["codebook_size"])
    rate_by_level = {level: codebook_rate(cfg, level) for level in range(1, levels + 1)}
    prompt_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for prompt_id in test_ids:
        x = collect_samples(cfg, tensor_name, layer, [prompt_id], part="eval").to(device=device, dtype=torch.float32)
        input_energy = (x * x).mean(dim=(1, 2)).clamp_min(1e-12)
        for control_name, cb_cpu in (("offline_rvq", codebooks_cpu), ("random_codebook", random_codebooks_cpu)):
            codebooks = cb_cpu.to(device=device, dtype=torch.float32)
            residual = x.clone()
            recon = torch.zeros_like(x)
            previous_mse = input_energy.clone()
            for level in range(levels):
                ids, ratio = assign_batched(residual, codebooks[:, level], return_ratio=True)
                assigned = gather_centroids(codebooks[:, level], ids)
                recon += assigned
                residual -= assigned
                metrics = reconstruction_metrics(x, recon)
                usage = usage_metrics(ids, clusters)
                entropy_rank, stable_rank = effective_ranks(residual)
                centered = residual - residual.mean(dim=(1, 2), keepdim=True)
                variance = centered.pow(2).mean(dim=(1, 2)).clamp_min(1e-12)
                kurtosis = centered.pow(4).mean(dim=(1, 2)) / variance.pow(2) - 3.0
                current_mse = (residual * residual).mean(dim=(1, 2))
                for local_head, head in enumerate(heads):
                    base = {
                        "study_id": cfg["study_id"],
                        "hypothesis_id": "H1" if control_name != "offline_rvq" else "H1,H3",
                        "fold": fold,
                        "initialization_seed": seed,
                        "prompt_id": int(prompt_id),
                        "generation_id": f"prompt_{prompt_id}_seed_{cfg['seed']}",
                        "source_chunk": "all_canonical",
                        "read_chunk": None,
                        "history_age": None,
                        "layer": layer,
                        "head": head,
                        "K_or_V": tensor_name,
                        "hook_stage": "pre_rope" if tensor_name == "K" else "pre_prope",
                        "RVQ_stage": level + 1,
                        "method": control_name,
                        "codebook_id": f"fold{fold}_seed{seed}_{tensor_name}_L{layer}_H{head}_S{level+1}",
                        **rate_by_level[level + 1],
                    }
                    prompt_rows.append(
                        {
                            **base,
                            **{key: float(value[local_head].item()) for key, value in metrics.items()},
                        }
                    )
                    if control_name == "offline_rvq":
                        marginal = (previous_mse[local_head] - current_mse[local_head]) / max(
                            1.0, math.ceil(math.log2(clusters))
                        )
                        stage_rows.append(
                            {
                                **base,
                                "residual_energy_ratio": float((current_mse[local_head] / input_energy[local_head]).item()),
                                "marginal_gain_per_bit": float(marginal.item()),
                                "residual_effective_rank": float(entropy_rank[local_head].item()),
                                "residual_stable_rank": float(stable_rank[local_head].item()),
                                "residual_excess_kurtosis": float(kurtosis[local_head].item()),
                                "codebook_entropy_bits": float(usage["entropy_bits"][local_head].item()),
                                "codebook_perplexity": float(usage["perplexity"][local_head].item()),
                                "dead_code_ratio": float(usage["dead_code_ratio"][local_head].item()),
                                "top_code_frequency": float(usage["top_code_frequency"][local_head].item()),
                                "nearest_first_second_distance_ratio": float(ratio[local_head].mean().item()),
                            }
                        )
                previous_mse = current_mse
    return prompt_rows, stage_rows


def run_cv_task(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str, layer: int, device: torch.device
) -> None:
    paths = cv_task_paths(output_root(cfg), fold, seed, tensor_name, layer)
    if all(paths[key].exists() for key in ("codebook", "metrics", "stage", "done")):
        return
    train_ids, test_ids = fold_ids(cfg, fold)
    x_cpu = collect_samples(cfg, tensor_name, layer, train_ids, part="train")
    x = x_cpu.to(device=device, dtype=torch.float32)
    levels = int(cfg["rvq"]["levels"])
    clusters = int(cfg["rvq"]["codebook_size"])
    iterations = int(cfg["rvq"]["kmeans_iterations"])
    started = time.time()
    codebooks, train_energy = fit_rvq_batched(
        x, levels=levels, clusters=clusters, iterations=iterations,
        seed=seed + fold * 10_000_019 + layer * 1009 + (0 if tensor_name == "K" else 500_009),
    )
    random_codebooks = random_rvq_codebooks(
        x, levels=levels, clusters=clusters,
        seed=seed + fold * 1_000_003 + layer * 917 + (0 if tensor_name == "K" else 700_001),
    )
    payload = {
        "schema_version": "wan5b_prompt_cv_rvq.v1",
        "config_hash": cfg_hash,
        "fold": fold,
        "initialization_seed": seed,
        "tensor": tensor_name,
        "hook_stage": "pre_rope" if tensor_name == "K" else "pre_prope",
        "layer": layer,
        "heads": torch.tensor(parse_axis(cfg["heads"], HEADS), dtype=torch.int16),
        "train_prompt_ids": train_ids,
        "test_prompt_ids": test_ids,
        "input_norm_mode": cfg["rvq"]["input_norm_mode"],
        "codebooks": codebooks,
        "random_codebooks": random_codebooks,
        "train_residual_mse": train_energy,
        "training_seconds": time.time() - started,
    }
    atomic_torch(paths["codebook"], payload)
    prompt_rows, stage_rows = evaluate_codebooks(
        cfg, fold, seed, tensor_name, layer, codebooks, random_codebooks, test_ids, device
    )
    for row in prompt_rows:
        row["config_hash"] = cfg_hash
    for row in stage_rows:
        row["config_hash"] = cfg_hash
    write_csv(paths["metrics"], prompt_rows)
    write_csv(paths["stage"], stage_rows)
    atomic_json(
        paths["done"],
        {
            "completed": utc_now(),
            "config_hash": cfg_hash,
            "fold": fold,
            "seed": seed,
            "tensor": tensor_name,
            "layer": layer,
            "prompt_metric_rows": len(prompt_rows),
            "stage_rows": len(stage_rows),
            "codebook_sha256": sha256_file(paths["codebook"]),
        },
    )
    print(
        f"[cv] fold={fold} seed={seed} tensor={tensor_name} layer={layer} "
        f"seconds={time.time()-started:.1f}", flush=True
    )


def cv_tasks(cfg: dict[str, Any]) -> list[tuple[int, int, str, int]]:
    return [
        (fold, int(seed), str(tensor_name), layer)
        for fold in range(int(cfg["folds"]))
        for seed in cfg["initialization_seeds"]
        for tensor_name in cfg["tensors"]
        for layer in parse_axis(cfg["layers"], LAYERS)
    ]


def cmd_cv_worker(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CV worker requires CUDA")
    device = torch.device("cuda:0")
    tasks = cv_tasks(cfg)
    selected = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_id]
    print(
        f"[cv-worker] shard={args.shard_id}/{args.num_shards} tasks={len(selected)} gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    for task in selected:
        run_cv_task(cfg, cfg_hash, *task, device)


def launch_workers(args: argparse.Namespace, worker_command: str) -> None:
    cfg_path = resolve_path(args.config)
    cfg, _cfg_hash = parse_config(cfg_path)
    configured = [int(value) for value in cfg["gpus"]]
    gpus = configured if not args.gpus else [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("No GPUs configured")
    root = output_root(cfg)
    log_dir = root / "logs" / worker_command
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard_id, gpu in enumerate(gpus):
        log_path = log_dir / f"gpu_{gpu}_shard_{shard_id}.log"
        handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            str(cfg["python"]), str(Path(__file__).resolve()), worker_command,
            "--config", str(cfg_path), "--shard-id", str(shard_id), "--num-shards", str(len(gpus)),
        ]
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((gpu, process, handle, log_path))
        print(f"[launch] gpu={gpu} pid={process.pid} log={log_path}", flush=True)
    last_report = 0.0
    try:
        while True:
            alive = [(gpu, process) for gpu, process, _handle, _path in processes if process.poll() is None]
            now = time.time()
            if now - last_report >= 10:
                status = ", ".join(
                    f"gpu{gpu}:pid{process.pid}:{'running' if process.poll() is None else process.returncode}"
                    for gpu, process, _handle, _path in processes
                )
                print(f"[monitor] {status}", flush=True)
                last_report = now
            if not alive:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        for _gpu, process, _handle, _path in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _gpu, _process, handle, _path in processes:
            handle.close()
    failed = [(gpu, process.returncode, path) for gpu, process, _handle, path in processes if process.returncode != 0]
    if failed:
        for gpu, returncode, path in failed:
            print(f"[failed] gpu={gpu} returncode={returncode} log={path}", flush=True)
            try:
                print(path.read_text(encoding="utf-8")[-8000:], flush=True)
            except Exception:
                pass
        raise RuntimeError(f"{len(failed)} {worker_command} workers failed")
    print(f"[launch] all {worker_command} workers completed", flush=True)


def merge_csv_glob(pattern: Iterable[Path], output: Path) -> int:
    paths = sorted(pattern)
    if not paths:
        raise FileNotFoundError(f"No partial CSV files found for {output}")
    fields: list[str] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        for field in header:
            if field not in fields:
                fields.append(field)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    count = 0
    with tmp.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for path in paths:
            with path.open("r", newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                for row in reader:
                    writer.writerow(row)
                    count += 1
    os.replace(tmp, output)
    return count


def cmd_merge_cv(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    root = output_root(cfg)
    expected = len(cv_tasks(cfg))
    done = list((root / "cv").glob("fold_*/seed_*/*/done/layer_*.json"))
    if len(done) != expected:
        raise RuntimeError(f"CV incomplete: expected {expected} done units, found {len(done)}")
    offline_rows = merge_csv_glob(
        (root / "cv").glob("fold_*/seed_*/*/partials/layer_*_prompt_metrics.csv"),
        root / "tables" / "offline_generalization.csv",
    )
    stage_rows = merge_csv_glob(
        (root / "cv").glob("fold_*/seed_*/*/partials/layer_*_stagewise.csv"),
        root / "tables" / "rvq_stagewise.csv",
    )
    codebook_manifest = []
    for path in sorted((root / "cv").glob("fold_*/seed_*/*/codebooks/layer_*.pt")):
        codebook_manifest.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    atomic_json(root / "cv" / "codebook_manifest.json", codebook_manifest)
    atomic_json(
        root / "cv" / "cv.done.json",
        {"completed": utc_now(), "config_hash": cfg_hash, "units": expected, "offline_rows": offline_rows, "stage_rows": stage_rows},
    )
    print(f"[merge-cv] units={expected} offline_rows={offline_rows} stage_rows={stage_rows}", flush=True)


def local_head_index(cfg: dict[str, Any], head: int) -> int:
    heads = parse_axis(cfg["heads"], HEADS)
    try:
        return heads.index(int(head))
    except ValueError as exc:
        raise ValueError(f"Head {head} is not enabled in config heads={heads}") from exc


def group_prompt_samples(
    cfg: dict[str, Any], tensor_name: str, layer: int, head: int, prompt_id: int, part: str
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = load_cached(cfg, prompt_id, layer)
    local_head = local_head_index(cfg, head)
    tensor = payload[tensor_name][local_head]
    chunks = payload["source_chunks"][local_head].long()
    train_take = int(cfg["sample_cache"]["train_samples_per_prompt_head"])
    eval_take = int(cfg["sample_cache"]["eval_samples_per_prompt_head"])
    if part == "train":
        return tensor[:train_take].float(), chunks[:train_take]
    if part == "eval":
        return tensor[-eval_take:].float(), chunks[-eval_take:]
    if part == "oracle_train":
        return tensor[:-eval_take].float(), chunks[:-eval_take]
    raise ValueError(part)


def load_local_codebook(
    cfg: dict[str, Any], fold: int, seed: int, tensor_name: str, layer: int, head: int
) -> tuple[torch.Tensor, torch.Tensor]:
    paths = cv_task_paths(output_root(cfg), fold, seed, tensor_name, layer)
    payload = torch.load(paths["codebook"], map_location="cpu", weights_only=True, mmap=True)
    local = local_head_index(cfg, head)
    return payload["codebooks"][local].float(), payload["random_codebooks"][local].float()


def decode_single_rvq(x: torch.Tensor, codebooks: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # ``Tensor.float()`` is a no-op when x is already FP32.  Clone before the
    # in-place residual updates so callers retain an immutable BF16/FP32
    # reference for reconstruction metrics and subsequent controls.
    residual = x.float().unsqueeze(0).clone()
    reconstruction = torch.zeros_like(residual)
    all_ids = []
    for level in range(codebooks.shape[0]):
        ids, _ = assign_batched(residual, codebooks[level].float().unsqueeze(0))
        reconstruction += gather_centroids(codebooks[level].float().unsqueeze(0), ids)
        residual -= gather_centroids(codebooks[level].float().unsqueeze(0), ids)
        all_ids.append(ids[0])
    return reconstruction[0], all_ids


def fit_single_rvq(
    x: torch.Tensor, cfg: dict[str, Any], seed: int, *, levels: int | None = None, clusters: int | None = None
) -> torch.Tensor:
    levels = int(levels if levels is not None else cfg["rvq"]["levels"])
    clusters = int(clusters if clusters is not None else cfg["rvq"]["codebook_size"])
    effective_clusters = min(clusters, int(x.shape[0]))
    cb, _energy = fit_rvq_batched(
        x.float().unsqueeze(0).cuda(),
        levels=levels,
        clusters=effective_clusters,
        iterations=int(cfg["rvq"]["kmeans_iterations"]),
        seed=seed,
    )
    return cb[0].float()


def scalar_rate(
    *, index_bits: float, metadata_bytes: float, amortization: int, method: str
) -> dict[str, Any]:
    total = float(index_bits) + float(metadata_bytes) * 8.0 / max(1, amortization)
    return {
        "method": method,
        "logical_index_bits_per_vector": float(index_bits),
        "index_bits_per_scalar": float(index_bits) / DIM,
        "metadata_bytes_per_group": float(metadata_bytes),
        "total_amortized_bits_per_vector": total,
        "total_amortized_bits_per_scalar": total / DIM,
    }


def uniform_quantize(
    x: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor, bits: int
) -> torch.Tensor:
    levels = max(2, 1 << int(bits))
    span = (maximum - minimum).clamp_min(1e-8)
    code = torch.round((x - minimum) / span * (levels - 1)).clamp(0, levels - 1)
    return minimum + code * span / (levels - 1)


def method_metric_row(
    cfg: dict[str, Any], cfg_hash: str, *, hypothesis: str, fold: int, seed: int,
    prompt_id: int, tensor_name: str, layer: int, head: int, method: str,
    x: torch.Tensor, recon: torch.Tensor, rate: dict[str, Any], extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = reconstruction_metrics(x.unsqueeze(0), recon.unsqueeze(0))
    return {
        "study_id": cfg["study_id"],
        "config_hash": cfg_hash,
        "hypothesis_id": hypothesis,
        "fold": fold,
        "initialization_seed": seed,
        "prompt_id": prompt_id,
        "layer": layer,
        "head": head,
        "K_or_V": tensor_name,
        "hook_stage": "pre_rope" if tensor_name == "K" else "pre_prope",
        "method": method,
        **rate,
        **{key: float(value[0].item()) for key, value in metrics.items()},
        **(extra or {}),
    }


def independently_permute_channels(x: torch.Tensor, seed: int) -> torch.Tensor:
    result = torch.empty_like(x)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for channel in range(x.shape[1]):
        order = torch.randperm(x.shape[0], generator=generator)
        result[:, channel] = x[order, channel]
    return result


def fit_pq(x: torch.Tensor, cfg: dict[str, Any], seed: int, subspaces: int) -> torch.Tensor:
    if DIM % subspaces != 0:
        raise ValueError(f"DIM={DIM} not divisible by PQ subspaces={subspaces}")
    subdim = DIM // subspaces
    grouped = x.reshape(x.shape[0], subspaces, subdim).permute(1, 0, 2).contiguous().cuda()
    codebooks, _energy = fit_rvq_batched(
        grouped,
        levels=1,
        clusters=int(cfg["rvq"]["codebook_size"]),
        iterations=int(cfg["rvq"]["kmeans_iterations"]),
        seed=seed,
    )
    return codebooks[:, 0].float()


def decode_pq(x: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    subspaces, _clusters, subdim = codebooks.shape
    grouped = x.reshape(x.shape[0], subspaces, subdim).permute(1, 0, 2).contiguous()
    ids, _ = assign_batched(grouped, codebooks)
    reconstructed = gather_centroids(codebooks, ids)
    return reconstructed.permute(1, 0, 2).reshape_as(x)


def h1_oracle_and_calibration(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    layer: int, head: int, device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_ids, test_ids = fold_ids(cfg, fold)
    offline_cb, random_cb = load_local_codebook(cfg, fold, seed, tensor_name, layer, head)
    offline_cb = offline_cb.to(device)
    random_cb = random_cb.to(device)
    oracle_rows = []
    for prompt_id in test_ids:
        eval_x, eval_chunks = group_prompt_samples(cfg, tensor_name, layer, head, prompt_id, "eval")
        oracle_train, oracle_chunks = group_prompt_samples(cfg, tensor_name, layer, head, prompt_id, "oracle_train")
        eval_device = eval_x.to(device)
        offline_recon, _ = decode_single_rvq(eval_device, offline_cb)
        random_recon, _ = decode_single_rvq(eval_device, random_cb)
        prompt_cb = fit_single_rvq(
            oracle_train, cfg,
            seed + fold * 10_000 + prompt_id * 101 + layer * 17 + head,
        ).to(device)
        prompt_recon, _ = decode_single_rvq(eval_device, prompt_cb)

        chunk_recon = torch.empty_like(eval_device)
        chunk_cluster_counts = []
        for chunk in sorted(set(int(value) for value in eval_chunks.tolist())):
            train_mask = oracle_chunks == chunk
            eval_mask = eval_chunks == chunk
            chunk_train = oracle_train[train_mask]
            if not chunk_train.numel():
                continue
            effective_clusters = min(int(cfg["rvq"]["codebook_size"]), int(chunk_train.shape[0]))
            chunk_cb = fit_single_rvq(
                chunk_train, cfg,
                seed + prompt_id * 1009 + chunk * 1_000_003 + layer * 31 + head,
                clusters=effective_clusters,
            ).to(device)
            chunk_recon[eval_mask], _ = decode_single_rvq(eval_device[eval_mask], chunk_cb)
            chunk_cluster_counts.append(effective_clusters)
        methods = [
            ("global_offline", offline_recon, {}),
            ("random_codebook", random_recon, {}),
            ("per_prompt_oracle", prompt_recon, {"oracle_training_vectors": int(oracle_train.shape[0])}),
            (
                "per_source_chunk_oracle", chunk_recon,
                {
                    "oracle_training_vectors": int(oracle_train.shape[0]),
                    "effective_codebook_size_min": min(chunk_cluster_counts) if chunk_cluster_counts else 0,
                    "effective_codebook_size_max": max(chunk_cluster_counts) if chunk_cluster_counts else 0,
                    "smoke_rate_comparable": min(chunk_cluster_counts or [0]) == int(cfg["rvq"]["codebook_size"]),
                },
            ),
        ]
        rate = codebook_rate(cfg, int(cfg["rvq"]["levels"]))
        for method, reconstruction, extra in methods:
            oracle_rows.append(
                method_metric_row(
                    cfg, cfg_hash, hypothesis="H1", fold=fold, seed=seed, prompt_id=prompt_id,
                    tensor_name=tensor_name, layer=layer, head=head, method=method,
                    x=eval_device, recon=reconstruction, rate=rate, extra=extra,
                )
            )

    calibration_rows = []
    sizes = [int(value) for value in cfg["h1"]["calibration_sizes"]]
    repeats = int(cfg["h1"]["calibration_repeats"])
    for size in sizes:
        if size > len(train_ids):
            continue
        for repeat in range(repeats):
            generator = torch.Generator(device="cpu").manual_seed(
                seed + fold * 100_003 + size * 1009 + repeat * 17 + layer * 7 + head
            )
            order = torch.randperm(len(train_ids), generator=generator).tolist()
            selected = sorted(train_ids[index] for index in order[:size])
            x_train = torch.cat(
                [group_prompt_samples(cfg, tensor_name, layer, head, pid, "train")[0] for pid in selected], dim=0
            )
            calibration_cb = fit_single_rvq(
                x_train, cfg, seed + size * 1_000_003 + repeat * 1009 + layer * 31 + head
            ).to(device)
            for prompt_id in test_ids:
                eval_x, _ = group_prompt_samples(cfg, tensor_name, layer, head, prompt_id, "eval")
                eval_device = eval_x.to(device)
                recon, _ = decode_single_rvq(eval_device, calibration_cb)
                calibration_rows.append(
                    method_metric_row(
                        cfg, cfg_hash, hypothesis="H1", fold=fold, seed=seed, prompt_id=prompt_id,
                        tensor_name=tensor_name, layer=layer, head=head, method="global_offline_calibration_curve",
                        x=eval_device, recon=recon,
                        rate=codebook_rate(cfg, int(cfg["rvq"]["levels"])),
                        extra={"calibration_prompt_count": size, "calibration_repeat": repeat, "calibration_prompt_ids": ";".join(map(str, selected))},
                    )
                )
    return oracle_rows, calibration_rows


def h2_equal_rate(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    layer: int, head: int, device: torch.device,
) -> list[dict[str, Any]]:
    train_ids, test_ids = fold_ids(cfg, fold)
    train = torch.cat(
        [group_prompt_samples(cfg, tensor_name, layer, head, pid, "train")[0] for pid in train_ids], dim=0
    ).float()
    local_cb, random_cb = load_local_codebook(cfg, fold, seed, tensor_name, layer, head)
    local_cb = local_cb.to(device)
    random_cb = random_cb.to(device)
    clusters = int(cfg["rvq"]["codebook_size"])
    levels = int(cfg["rvq"]["levels"])
    rvq_index_bits = levels * math.ceil(math.log2(clusters))
    amort = int(cfg["rvq"]["amortization_vectors_per_group"])

    pq_m = int(cfg["h2"]["pq_subspaces"])
    pq_cb = fit_pq(train, cfg, seed + fold * 1009 + layer * 31 + head, pq_m).to(device)
    vq_cb = fit_single_rvq(train, cfg, seed + 700_001 + layer * 17 + head, levels=1).to(device)

    mean = train.mean(dim=0)
    centered = train - mean
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    pca_rank = min(int(cfg["h2"]["pca_components"]), DIM)
    basis = eigenvectors[:, :pca_rank]
    train_coeff = centered @ basis

    active_bits = max(1, min(8, rvq_index_bits // max(1, pca_rank)))
    active_channels = min(DIM, rvq_index_bits // active_bits)
    variance_order = torch.argsort(train.var(dim=0), descending=True)
    active_idx = variance_order[:active_channels]
    active_min = train[:, active_idx].amin(dim=0)
    active_max = train[:, active_idx].amax(dim=0)

    coeff_min = train_coeff.amin(dim=0)
    coeff_max = train_coeff.amax(dim=0)
    whiten_scale = torch.sqrt(eigenvalues[:pca_rank].clamp_min(1e-8))
    whiten_train = train_coeff / whiten_scale
    whiten_min = whiten_train.amin(dim=0)
    whiten_max = whiten_train.amax(dim=0)

    permuted_train = independently_permute_channels(train, seed + fold * 100_003 + layer * 1009 + head)
    permuted_cb = fit_single_rvq(permuted_train, cfg, seed + 900_001 + layer * 31 + head).to(device)

    result = []
    for prompt_id in test_ids:
        eval_x, _ = group_prompt_samples(cfg, tensor_name, layer, head, prompt_id, "eval")
        x = eval_x.to(device)
        offline_recon, _ = decode_single_rvq(x, local_cb)
        random_recon, _ = decode_single_rvq(x, random_cb)
        pq_recon = decode_pq(x, pq_cb)
        vq_recon, _ = decode_single_rvq(x, vq_cb)
        methods: list[tuple[str, torch.Tensor, dict[str, Any], dict[str, Any]]] = [
            ("rvq", offline_recon, codebook_rate(cfg, levels), {"joint_structure": "intact"}),
            ("random_codebook", random_recon, codebook_rate(cfg, levels), {"joint_structure": "intact"}),
            (
                "pq", pq_recon,
                scalar_rate(
                    index_bits=pq_m * math.ceil(math.log2(clusters)),
                    metadata_bytes=(clusters * DIM / 2 + pq_m * clusters * 2),
                    amortization=amort, method="pq",
                ),
                {"joint_structure": "subspace", "pq_subspaces": pq_m},
            ),
            (
                "single_stage_vq", vq_recon,
                scalar_rate(
                    index_bits=math.ceil(math.log2(clusters)),
                    metadata_bytes=(clusters * DIM / 2 + clusters * 2),
                    amortization=amort, method="single_stage_vq",
                ),
                {"joint_structure": "intact"},
            ),
        ]

        active_recon = mean.to(device).expand_as(x).clone()
        active_recon[:, active_idx.to(device)] = uniform_quantize(
            x[:, active_idx.to(device)], active_min.to(device), active_max.to(device), active_bits
        )
        methods.append(
            (
                "equal_index_active_channel_scalar", active_recon,
                scalar_rate(
                    index_bits=active_channels * active_bits,
                    metadata_bytes=active_channels * 8 + DIM * 4,
                    amortization=amort, method="equal_index_active_channel_scalar",
                ),
                {"joint_structure": "destroyed_by_scalar", "scalar_bits": active_bits, "active_channels": active_channels},
            )
        )

        coeff = (x.cpu() - mean) @ basis
        coeff_hat = uniform_quantize(coeff, coeff_min, coeff_max, active_bits)
        pca_recon = (mean + coeff_hat @ basis.T).to(device)
        methods.append(
            (
                "pca_scalar", pca_recon,
                scalar_rate(
                    index_bits=pca_rank * active_bits,
                    metadata_bytes=(DIM + DIM * pca_rank + 2 * pca_rank) * 4,
                    amortization=amort, method="pca_scalar",
                ),
                {"joint_structure": "linear_transform", "scalar_bits": active_bits, "pca_rank": pca_rank},
            )
        )
        whiten = coeff / whiten_scale
        whiten_hat = uniform_quantize(whiten, whiten_min, whiten_max, active_bits)
        whiten_recon = (mean + (whiten_hat * whiten_scale) @ basis.T).to(device)
        methods.append(
            (
                "whitening_scalar", whiten_recon,
                scalar_rate(
                    index_bits=pca_rank * active_bits,
                    metadata_bytes=(DIM + DIM * pca_rank + 3 * pca_rank) * 4,
                    amortization=amort, method="whitening_scalar",
                ),
                {"joint_structure": "whitened", "scalar_bits": active_bits, "pca_rank": pca_rank},
            )
        )
        for bits in [int(value) for value in cfg["h2"]["scalar_bits"]]:
            max_abs = train.abs().max().to(device)
            sym_recon = uniform_quantize(x, -max_abs, max_abs, bits)
            methods.append(
                (
                    f"symmetric_scalar_{bits}bit", sym_recon,
                    scalar_rate(index_bits=DIM * bits, metadata_bytes=4, amortization=amort, method=f"symmetric_scalar_{bits}bit"),
                    {"joint_structure": "destroyed_by_scalar", "scalar_bits": bits},
                )
            )
            channel_min = train.amin(dim=0).to(device)
            channel_max = train.amax(dim=0).to(device)
            pc_recon = uniform_quantize(x, channel_min, channel_max, bits)
            methods.append(
                (
                    f"per_channel_asymmetric_scalar_{bits}bit", pc_recon,
                    scalar_rate(index_bits=DIM * bits, metadata_bytes=DIM * 8, amortization=amort, method=f"per_channel_asymmetric_scalar_{bits}bit"),
                    {"joint_structure": "destroyed_by_scalar", "scalar_bits": bits},
                )
            )

        for method, recon, rate, extra in methods:
            result.append(
                method_metric_row(
                    cfg, cfg_hash, hypothesis="H2", fold=fold, seed=seed, prompt_id=prompt_id,
                    tensor_name=tensor_name, layer=layer, head=head, method=method,
                    x=x, recon=recon, rate=rate, extra=extra,
                )
            )

        perm_eval = independently_permute_channels(
            eval_x, seed + fold * 1_000_003 + prompt_id * 1009 + layer * 31 + head
        ).to(device)
        perm_recon, _ = decode_single_rvq(perm_eval, permuted_cb)
        result.append(
            method_metric_row(
                cfg, cfg_hash, hypothesis="H2", fold=fold, seed=seed, prompt_id=prompt_id,
                tensor_name=tensor_name, layer=layer, head=head, method="rvq_independent_channel_permutation",
                x=perm_eval, recon=perm_recon, rate=codebook_rate(cfg, levels),
                extra={"joint_structure": "independently_permuted_per_channel"},
            )
        )
    return result


def secondary_task_paths(root: Path, fold: int, seed: int, tensor_name: str, layer: int, head: int) -> dict[str, Path]:
    base = root / "secondary" / f"fold_{fold}" / f"seed_{seed}" / tensor_name.lower()
    stem = f"layer_{layer:02d}_head_{head:02d}"
    return {
        "oracle": base / "h1_oracles" / f"{stem}.csv",
        "calibration": base / "h1_calibration" / f"{stem}.csv",
        "h2": base / "h2_equal_rate" / f"{stem}.csv",
        "done": base / "done" / f"{stem}.json",
    }


def secondary_tasks(cfg: dict[str, Any]) -> list[tuple[int, int, str, int, int]]:
    configured = {(int(layer), int(head)) for layer, head in cfg["h1"]["oracle_groups"]}
    h2_groups = {
        (int(layer), int(head))
        for layer in cfg["h2"]["representative_layers"]
        for head in cfg["h2"]["representative_heads"]
    }
    groups = sorted(configured | h2_groups)
    enabled_layers = set(parse_axis(cfg["layers"], LAYERS))
    enabled_heads = set(parse_axis(cfg["heads"], HEADS))
    groups = [(layer, head) for layer, head in groups if layer in enabled_layers and head in enabled_heads]
    return [
        (fold, int(seed), str(tensor_name), layer, head)
        for fold in range(int(cfg["folds"]))
        for seed in cfg["initialization_seeds"]
        for tensor_name in cfg["tensors"]
        for layer, head in groups
    ]


def run_secondary_task(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    layer: int, head: int, device: torch.device,
) -> None:
    paths = secondary_task_paths(output_root(cfg), fold, seed, tensor_name, layer, head)
    if all(path.exists() for path in paths.values()):
        return
    started = time.time()
    oracle_rows, calibration_rows = h1_oracle_and_calibration(
        cfg, cfg_hash, fold, seed, tensor_name, layer, head, device
    )
    h2_rows = h2_equal_rate(cfg, cfg_hash, fold, seed, tensor_name, layer, head, device)
    write_csv(paths["oracle"], oracle_rows)
    write_csv(paths["calibration"], calibration_rows)
    write_csv(paths["h2"], h2_rows)
    atomic_json(
        paths["done"],
        {
            "completed": utc_now(), "config_hash": cfg_hash, "fold": fold, "seed": seed,
            "tensor": tensor_name, "layer": layer, "head": head,
            "oracle_rows": len(oracle_rows), "calibration_rows": len(calibration_rows), "h2_rows": len(h2_rows),
        },
    )
    print(
        f"[secondary] fold={fold} seed={seed} tensor={tensor_name} L={layer} H={head} "
        f"seconds={time.time()-started:.1f}", flush=True
    )


def cmd_secondary_worker(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Secondary worker requires CUDA")
    tasks = secondary_tasks(cfg)
    selected = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_id]
    print(f"[secondary-worker] shard={args.shard_id}/{args.num_shards} tasks={len(selected)}", flush=True)
    for task in selected:
        run_secondary_task(cfg, cfg_hash, *task, torch.device("cuda:0"))


def cmd_merge_secondary(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    root = output_root(cfg)
    expected = len(secondary_tasks(cfg))
    done = list((root / "secondary").glob("fold_*/seed_*/*/done/*.json"))
    if len(done) != expected:
        raise RuntimeError(f"Secondary incomplete: expected {expected}, found {len(done)}")
    oracle = merge_csv_glob(
        (root / "secondary").glob("fold_*/seed_*/*/h1_oracles/*.csv"),
        root / "tables" / "offline_oracles.csv",
    )
    calibration = merge_csv_glob(
        (root / "secondary").glob("fold_*/seed_*/*/h1_calibration/*.csv"),
        root / "tables" / "calibration_size_curve.csv",
    )
    h2 = merge_csv_glob(
        (root / "secondary").glob("fold_*/seed_*/*/h2_equal_rate/*.csv"),
        root / "tables" / "equal_rate_quantization_index_budget.csv",
    )
    atomic_json(
        root / "secondary" / "secondary.done.json",
        {"completed": utc_now(), "config_hash": cfg_hash, "units": expected, "oracle_rows": oracle, "calibration_rows": calibration, "h2_rows": h2},
    )
    print(f"[merge-secondary] units={expected} oracle={oracle} calibration={calibration} h2={h2}", flush=True)


def fit_pq_clusters(
    x: torch.Tensor, cfg: dict[str, Any], seed: int, subspaces: int, clusters: int
) -> torch.Tensor:
    if DIM % subspaces != 0:
        raise ValueError(f"DIM={DIM} not divisible by PQ subspaces={subspaces}")
    subdim = DIM // subspaces
    grouped = x.reshape(x.shape[0], subspaces, subdim).permute(1, 0, 2).contiguous().cuda()
    codebooks, _energy = fit_rvq_batched(
        grouped, levels=1, clusters=int(clusters),
        iterations=int(cfg["rvq"]["kmeans_iterations"]), seed=seed,
    )
    return codebooks[:, 0].float()


def h2_sweep_task_paths(
    root: Path, fold: int, seed: int, tensor_name: str, layer: int, head: int
) -> dict[str, Path]:
    base = root / "h2_total_rate_sweep" / f"fold_{fold}" / f"seed_{seed}" / tensor_name.lower()
    stem = f"layer_{layer:02d}_head_{head:02d}"
    return {"metrics": base / "partials" / f"{stem}.csv", "done": base / "done" / f"{stem}.json"}


def run_h2_sweep_task(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    layer: int, head: int, device: torch.device,
) -> None:
    paths = h2_sweep_task_paths(output_root(cfg), fold, seed, tensor_name, layer, head)
    if all(path.exists() for path in paths.values()):
        return
    started = time.time()
    train_ids, test_ids = fold_ids(cfg, fold)
    train = torch.cat(
        [group_prompt_samples(cfg, tensor_name, layer, head, pid, "train")[0] for pid in train_ids], dim=0
    ).float()
    base_clusters = int(cfg["rvq"]["codebook_size"])
    total_clusters = base_clusters * 2
    subspaces = int(cfg["h2"]["pq_subspaces"])
    pq = fit_pq_clusters(
        train, cfg, seed + fold * 1009 + layer * 31 + head + 8_000_003,
        subspaces, total_clusters,
    ).to(device)
    mean = train.mean(dim=0)
    centered = train - mean
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    pca_rank = min(int(cfg["h2"]["pca_components"]), DIM)
    basis = eigenvectors[:, order[:pca_rank]]
    train_coeff = centered @ basis
    coefficient_bits = max(
        1,
        int(cfg["rvq"]["levels"]) * math.ceil(math.log2(base_clusters)) // pca_rank + 1,
    )
    coeff_min = train_coeff.amin(dim=0)
    coeff_max = train_coeff.amax(dim=0)
    amort = int(cfg["rvq"]["amortization_vectors_per_group"])
    rows = []
    for prompt_id in test_ids:
        eval_x, _ = group_prompt_samples(cfg, tensor_name, layer, head, prompt_id, "eval")
        x = eval_x.to(device)
        pq_recon = decode_pq(x, pq)
        pq_rate = scalar_rate(
            index_bits=subspaces * math.ceil(math.log2(total_clusters)),
            metadata_bytes=(total_clusters * DIM / 2 + subspaces * total_clusters * 2),
            amortization=amort,
            method=f"pq_c{total_clusters}_total_rate",
        )
        rows.append(
            method_metric_row(
                cfg, cfg_hash, hypothesis="H2", fold=fold, seed=seed, prompt_id=prompt_id,
                tensor_name=tensor_name, layer=layer, head=head,
                method=f"pq_c{total_clusters}_total_rate", x=x, recon=pq_recon, rate=pq_rate,
                extra={"joint_structure": "subspace", "pq_subspaces": subspaces,
                       "rate_matching_scope": "total_amortized_sweep_candidate"},
            )
        )
        coeff = (eval_x.float() - mean) @ basis
        coeff_hat = uniform_quantize(coeff, coeff_min, coeff_max, coefficient_bits)
        pca_recon = (mean + coeff_hat @ basis.T).to(device)
        pca_rate = scalar_rate(
            index_bits=pca_rank * coefficient_bits,
            metadata_bytes=(DIM + DIM * pca_rank + 2 * pca_rank) * 4,
            amortization=amort,
            method=f"pca_scalar_{coefficient_bits}bit_total_rate",
        )
        rows.append(
            method_metric_row(
                cfg, cfg_hash, hypothesis="H2", fold=fold, seed=seed, prompt_id=prompt_id,
                tensor_name=tensor_name, layer=layer, head=head,
                method=f"pca_scalar_{coefficient_bits}bit_total_rate", x=x, recon=pca_recon,
                rate=pca_rate,
                extra={"joint_structure": "linear_transform", "scalar_bits": coefficient_bits,
                       "pca_rank": pca_rank, "rate_matching_scope": "total_amortized_sweep_candidate"},
            )
        )
    write_csv(paths["metrics"], rows)
    atomic_json(
        paths["done"],
        {"completed": utc_now(), "config_hash": cfg_hash, "fold": fold, "seed": seed,
         "tensor": tensor_name, "layer": layer, "head": head, "rows": len(rows),
         "seconds": time.time() - started},
    )
    print(
        f"[h2-sweep] fold={fold} seed={seed} tensor={tensor_name} L={layer} H={head} "
        f"seconds={time.time()-started:.1f}", flush=True,
    )


def cmd_h2_sweep_worker(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    if not torch.cuda.is_available():
        raise RuntimeError("H2 total-rate sweep requires CUDA")
    tasks = secondary_tasks(cfg)
    selected = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_id]
    print(f"[h2-sweep-worker] shard={args.shard_id}/{args.num_shards} tasks={len(selected)}", flush=True)
    for task in selected:
        run_h2_sweep_task(cfg, cfg_hash, *task, torch.device("cuda:0"))


def cmd_merge_h2_sweep(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    root = output_root(cfg)
    expected = len(secondary_tasks(cfg))
    done = list((root / "h2_total_rate_sweep").glob("fold_*/seed_*/*/done/*.json"))
    if len(done) != expected:
        raise RuntimeError(f"H2 total-rate sweep incomplete: expected {expected}, found {len(done)}")
    sweep_path = root / "tables" / "equal_rate_quantization_total_rate_sweep.csv"
    sweep_rows = merge_csv_glob(
        (root / "h2_total_rate_sweep").glob("fold_*/seed_*/*/partials/*.csv"), sweep_path,
    )
    index_path = root / "tables" / "equal_rate_quantization_index_budget.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Run merge-secondary first: {index_path}")
    combined_rows = merge_csv_glob(
        [index_path, sweep_path], root / "tables" / "equal_rate_quantization.csv"
    )
    atomic_json(
        root / "h2_total_rate_sweep" / "h2_sweep.done.json",
        {"completed": utc_now(), "config_hash": cfg_hash, "units": expected,
         "sweep_rows": sweep_rows, "combined_rows": combined_rows},
    )
    print(f"[merge-h2-sweep] units={expected} sweep={sweep_rows} combined={combined_rows}", flush=True)


def all_groups(cfg: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (layer, head)
        for layer in parse_axis(cfg["layers"], LAYERS)
        for head in parse_axis(cfg["heads"], HEADS)
    ]


def collect_group_matrix(
    cfg: dict[str, Any], tensor_name: str, prompt_ids: Sequence[int], *, part: str
) -> torch.Tensor:
    layers = parse_axis(cfg["layers"], LAYERS)
    matrices = []
    for layer in layers:
        matrices.append(collect_samples(cfg, tensor_name, layer, prompt_ids, part=part))
    return torch.cat(matrices, dim=0).contiguous()


def load_all_local_codebooks(
    cfg: dict[str, Any], fold: int, seed: int, tensor_name: str
) -> torch.Tensor:
    result = []
    for layer in parse_axis(cfg["layers"], LAYERS):
        path = cv_task_paths(output_root(cfg), fold, seed, tensor_name, layer)["codebook"]
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        result.append(payload["codebooks"])
    return torch.cat(result, dim=0).float().contiguous()


def feature_cluster_ids(descriptors: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    groups = descriptors.shape[0]
    if count >= groups:
        return torch.arange(groups, dtype=torch.long)
    if count <= 1:
        return torch.zeros(groups, dtype=torch.long)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    values = descriptors.to(device=device, dtype=torch.float32).unsqueeze(0)
    _centroids, assigned, _counts = kmeans_batched(
        values, clusters=count, iterations=15, seed=seed
    )
    labels_tensor = assigned[0].detach().cpu()
    # Lloyd updates may retain an initialized-but-unused centroid.  Sharing
    # clusters must all own at least one layer-head, so deterministically move
    # one member from the largest donor cluster into each empty label.
    counts = torch.bincount(labels_tensor, minlength=count)
    for empty in torch.nonzero(counts == 0, as_tuple=False).flatten().tolist():
        donor = int(torch.argmax(counts).item())
        members = torch.nonzero(labels_tensor == donor, as_tuple=False).flatten()
        if members.numel() <= 1:
            raise RuntimeError("Cannot repair an empty H4 descriptor cluster")
        moved = int(members[-1].item())
        labels_tensor[moved] = int(empty)
        counts[donor] -= 1
        counts[empty] += 1
    labels = labels_tensor.numpy()
    # KMeans label numbers are arbitrary. Canonicalize by smallest member ID.
    members = sorted(
        ((int(label), int(np.flatnonzero(labels == label).min())) for label in set(labels)),
        key=lambda item: item[1],
    )
    mapping = {old: new for new, (old, _first) in enumerate(members)}
    return torch.tensor([mapping[int(value)] for value in labels], dtype=torch.long)


def random_cluster_ids(groups: int, count: int, seed: int) -> torch.Tensor:
    if count >= groups:
        return torch.arange(groups, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(groups, generator=generator)
    labels = torch.empty(groups, dtype=torch.long)
    for position, group in enumerate(order.tolist()):
        labels[group] = position % count
    return labels


def balanced_cluster_samples(
    train: torch.Tensor, labels: torch.Tensor, count: int, sample_count: int, seed: int
) -> torch.Tensor:
    clusters = []
    for cluster in range(count):
        members = torch.nonzero(labels == cluster, as_tuple=False).flatten().tolist()
        if not members:
            raise ValueError(f"Empty sharing cluster {cluster}/{count}")
        per_member = math.ceil(sample_count / len(members))
        pieces = []
        for member in members:
            generator = torch.Generator(device="cpu").manual_seed(
                int(seed) + cluster * 1_000_003 + member * 1009
            )
            take = min(per_member, train.shape[1])
            indices = torch.randperm(train.shape[1], generator=generator)[:take]
            pieces.append(train[member, indices])
        values = torch.cat(pieces, dim=0)
        if values.shape[0] < sample_count:
            repeats = math.ceil(sample_count / values.shape[0])
            values = values.repeat((repeats, 1))
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + cluster * 97 + 13)
        keep = torch.randperm(values.shape[0], generator=generator)[:sample_count]
        clusters.append(values[keep])
    return torch.stack(clusters, dim=0).contiguous()


def fit_shared_codebooks(
    cfg: dict[str, Any], train: torch.Tensor, labels: torch.Tensor, count: int, seed: int, device: torch.device
) -> torch.Tensor:
    samples = balanced_cluster_samples(
        train, labels, count, int(cfg["h4"]["shared_train_samples"]), seed
    )
    outputs = []
    batch_groups = 24
    for start in range(0, count, batch_groups):
        end = min(count, start + batch_groups)
        codebooks, _energy = fit_rvq_batched(
            samples[start:end].to(device=device, dtype=torch.float32),
            levels=int(cfg["rvq"]["levels"]),
            clusters=int(cfg["rvq"]["codebook_size"]),
            iterations=int(cfg["rvq"]["kmeans_iterations"]),
            seed=seed + start * 1009,
        )
        outputs.append(codebooks)
    return torch.cat(outputs, dim=0).float().contiguous()


def decode_grouped_rvq(
    x: torch.Tensor, codebooks: torch.Tensor, levels: int | None = None
) -> torch.Tensor:
    levels = int(levels if levels is not None else codebooks.shape[1])
    residual = x.float().clone()
    reconstruction = torch.zeros_like(residual)
    for level in range(levels):
        ids, _ = assign_batched(residual, codebooks[:, level])
        assigned = gather_centroids(codebooks[:, level], ids)
        reconstruction += assigned
        residual -= assigned
    return reconstruction


def sharing_rate(cfg: dict[str, Any], levels: int, codebook_count: int, groups: int) -> dict[str, float]:
    base = codebook_rate(cfg, levels)
    logical = base["logical_index_bits_per_vector"]
    shared_bytes_per_target = base["codebook_storage_bytes_per_group"] * codebook_count / groups
    amort = int(cfg["rvq"]["amortization_vectors_per_group"])
    total = logical + shared_bytes_per_target * 8 / amort
    return {
        "logical_index_bits_per_vector": logical,
        "index_bits_per_scalar": logical / DIM,
        "total_codebook_count": codebook_count,
        "local_group_count": groups,
        "codebook_storage_bytes_total": base["codebook_storage_bytes_per_group"] * codebook_count,
        "total_amortized_bits_per_vector": total,
        "total_amortized_bits_per_scalar": total / DIM,
    }


def evaluate_sharing_count(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    labels: torch.Tensor, codebooks: torch.Tensor, count: int, policy: str,
    test_ids: Sequence[int], device: torch.device,
) -> list[dict[str, Any]]:
    groups = all_groups(cfg)
    group_codebooks = codebooks[labels].contiguous()
    result = []
    levels = int(cfg["rvq"]["levels"])
    rate = sharing_rate(cfg, levels, count, len(groups))
    for prompt_id in test_ids:
        x_cpu = collect_group_matrix(cfg, tensor_name, [prompt_id], part="eval")
        for start in range(0, len(groups), 24):
            end = min(len(groups), start + 24)
            x = x_cpu[start:end].to(device=device, dtype=torch.float32)
            cb = group_codebooks[start:end].to(device=device, dtype=torch.float32)
            recon = decode_grouped_rvq(x, cb)
            metrics = reconstruction_metrics(x, recon)
            for local, (layer, head) in enumerate(groups[start:end]):
                result.append(
                    {
                        "study_id": cfg["study_id"], "config_hash": cfg_hash, "hypothesis_id": "H4",
                        "fold": fold, "initialization_seed": seed, "prompt_id": prompt_id,
                        "layer": layer, "head": head, "K_or_V": tensor_name,
                        "hook_stage": "pre_rope" if tensor_name == "K" else "pre_prope",
                        "method": "shared_rvq", "grouping_policy": policy,
                        "sharing_cluster": int(labels[start + local]),
                        **rate,
                        **{key: float(value[local].item()) for key, value in metrics.items()},
                    }
                )
    return result


def structured_transfer_pairs(groups: Sequence[tuple[int, int]], modes: Sequence[str]) -> list[tuple[int, int, str]]:
    relationships: dict[tuple[int, int], set[str]] = defaultdict(set)
    for target, (target_layer, target_head) in enumerate(groups):
        for source, (source_layer, source_head) in enumerate(groups):
            if source == target:
                relationships[(source, target)].add("self")
                continue
            if "same_layer" in modes and source_layer == target_layer:
                relationships[(source, target)].add("same_layer")
            if "same_head" in modes and source_head == target_head:
                relationships[(source, target)].add("same_head")
            if "adjacent_layer" in modes and source_head == target_head and abs(source_layer - target_layer) == 1:
                relationships[(source, target)].add("adjacent_layer")
    return [(source, target, ";".join(sorted(labels))) for (source, target), labels in sorted(relationships.items())]


def evaluate_structured_transfer(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str,
    local_codebooks: torch.Tensor, test_ids: Sequence[int], device: torch.device,
) -> list[dict[str, Any]]:
    groups = all_groups(cfg)
    pairs = structured_transfer_pairs(groups, cfg["h4"]["transfer_modes"])
    requested_levels = [int(value) for value in cfg["h4"]["transfer_levels"]]
    result = []
    for prompt_id in test_ids:
        eval_matrix = collect_group_matrix(cfg, tensor_name, [prompt_id], part="eval")
        # Baselines are evaluated once for each requested prefix.
        baselines: dict[int, torch.Tensor] = {}
        for level in requested_levels:
            values = []
            for start in range(0, len(groups), 24):
                end = min(len(groups), start + 24)
                x = eval_matrix[start:end].to(device=device, dtype=torch.float32)
                cb = local_codebooks[start:end].to(device=device, dtype=torch.float32)
                recon = decode_grouped_rvq(x, cb, levels=level)
                values.append(((recon - x) ** 2).mean(dim=(1, 2)).detach().cpu())
            baselines[level] = torch.cat(values).clamp_min(1e-12)

        # Group by target to reuse its held-out vectors across source codebooks.
        by_target: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for source, target, relationship in pairs:
            by_target[target].append((source, relationship))
        for target, candidates in by_target.items():
            x_one = eval_matrix[target].to(device=device, dtype=torch.float32)
            batch_size = 32
            for start in range(0, len(candidates), batch_size):
                selected = candidates[start : start + batch_size]
                source_ids = [source for source, _relationship in selected]
                x = x_one.unsqueeze(0).expand(len(selected), -1, -1)
                cb = local_codebooks[source_ids].to(device=device, dtype=torch.float32)
                for level in requested_levels:
                    recon = decode_grouped_rvq(x, cb, levels=level)
                    distortion = ((recon - x) ** 2).mean(dim=(1, 2)).detach().cpu()
                    for local, (source, relationship) in enumerate(selected):
                        source_layer, source_head = groups[source]
                        target_layer, target_head = groups[target]
                        result.append(
                            {
                                "study_id": cfg["study_id"], "config_hash": cfg_hash, "hypothesis_id": "H4",
                                "fold": fold, "initialization_seed": seed, "prompt_id": prompt_id,
                                "K_or_V": tensor_name, "hook_stage": "pre_rope" if tensor_name == "K" else "pre_prope",
                                "RVQ_stage": level, "source_group": source, "target_group": target,
                                "source_layer": source_layer, "source_head": source_head,
                                "target_layer": target_layer, "target_head": target_head,
                                "relationship": relationship,
                                "source_distortion_mse": float(distortion[local].item()),
                                "target_local_distortion_mse": float(baselines[level][target].item()),
                                "transfer_ratio": float((distortion[local] / baselines[level][target]).item()),
                            }
                        )
    return result


def h4_task_paths(root: Path, fold: int, seed: int, tensor_name: str) -> dict[str, Path]:
    base = root / "h4" / f"fold_{fold}" / f"seed_{seed}" / tensor_name.lower()
    return {
        "scaling": base / "codebook_count_scaling.csv",
        "transfer": base / "codebook_transfer_matrix.csv",
        "assignments": base / "sharing_assignments.json",
        "done": base / "done.json",
    }


def h4_tasks(cfg: dict[str, Any]) -> list[tuple[int, int, str]]:
    return [
        (fold, int(seed), str(tensor_name))
        for fold in range(int(cfg["folds"]))
        for seed in cfg["initialization_seeds"]
        for tensor_name in cfg["tensors"]
    ]


def run_h4_task(
    cfg: dict[str, Any], cfg_hash: str, fold: int, seed: int, tensor_name: str, device: torch.device
) -> None:
    paths = h4_task_paths(output_root(cfg), fold, seed, tensor_name)
    if all(path.exists() for path in paths.values()):
        return
    started = time.time()
    train_ids, test_ids = fold_ids(cfg, fold)
    train = collect_group_matrix(cfg, tensor_name, train_ids, part="train")
    groups = all_groups(cfg)
    local_codebooks = load_all_local_codebooks(cfg, fold, seed, tensor_name)
    mean = train.float().mean(dim=1)
    std = train.float().std(dim=1).clamp_min(1e-6)
    rms = torch.sqrt((train.float() ** 2).mean(dim=(1, 2))).log().unsqueeze(1)
    descriptors = torch.cat([mean, std.log(), rms], dim=1)
    descriptors = (descriptors - descriptors.mean(dim=0)) / descriptors.std(dim=0).clamp_min(1e-6)

    scaling_rows = []
    assignment_manifest: dict[str, Any] = {}
    counts = sorted(set(min(int(value), len(groups)) for value in cfg["h4"]["codebook_counts"]))
    for count in counts:
        labels = feature_cluster_ids(descriptors, count, seed + fold * 1009 + count * 31)
        if count == len(groups):
            codebooks = local_codebooks
        else:
            codebooks = fit_shared_codebooks(
                cfg, train, labels, count, seed + fold * 1_000_003 + count * 1009, device
            )
        scaling_rows.extend(
            evaluate_sharing_count(
                cfg, cfg_hash, fold, seed, tensor_name, labels, codebooks, count,
                "feature_kmeans", test_ids, device,
            )
        )
        assignment_manifest[f"feature_kmeans_{count}"] = labels.tolist()

    nontrivial = [count for count in counts if 1 < count < len(groups)]
    if nontrivial:
        random_count = min(nontrivial, key=lambda value: abs(value - math.sqrt(len(groups))))
        labels = random_cluster_ids(len(groups), random_count, seed + fold * 1009 + 71)
        codebooks = fit_shared_codebooks(
            cfg, train, labels, random_count, seed + fold * 1_000_003 + random_count * 1009 + 17, device
        )
        scaling_rows.extend(
            evaluate_sharing_count(
                cfg, cfg_hash, fold, seed, tensor_name, labels, codebooks, random_count,
                "random_balanced", test_ids, device,
            )
        )
        assignment_manifest[f"random_balanced_{random_count}"] = labels.tolist()

    transfer_rows = evaluate_structured_transfer(
        cfg, cfg_hash, fold, seed, tensor_name, local_codebooks, test_ids, device
    )
    write_csv(paths["scaling"], scaling_rows)
    write_csv(paths["transfer"], transfer_rows)
    atomic_json(paths["assignments"], {"groups": groups, "assignments": assignment_manifest})
    atomic_json(
        paths["done"],
        {
            "completed": utc_now(), "config_hash": cfg_hash, "fold": fold, "seed": seed,
            "tensor": tensor_name, "scaling_rows": len(scaling_rows), "transfer_rows": len(transfer_rows),
            "seconds": time.time() - started,
        },
    )
    print(f"[h4] fold={fold} seed={seed} tensor={tensor_name} seconds={time.time()-started:.1f}", flush=True)


def cmd_h4_worker(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    if not torch.cuda.is_available():
        raise RuntimeError("H4 worker requires CUDA")
    # Six H4 workers otherwise each inherit the host-wide OpenMP pool (about
    # 130 threads), creating hundreds of runnable threads for 128 CPUs.
    # Partition intra-op CPU capacity explicitly; GPU work remains one process
    # per selected physical card.
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // int(args.num_shards)))
    torch.set_num_interop_threads(1)
    tasks = h4_tasks(cfg)
    selected = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_id]
    print(f"[h4-worker] shard={args.shard_id}/{args.num_shards} tasks={len(selected)}", flush=True)
    for task in selected:
        run_h4_task(cfg, cfg_hash, *task, torch.device("cuda:0"))


def cmd_merge_h4(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    root = output_root(cfg)
    expected = len(h4_tasks(cfg))
    done = list((root / "h4").glob("fold_*/seed_*/*/done.json"))
    if len(done) != expected:
        raise RuntimeError(f"H4 incomplete: expected {expected}, found {len(done)}")
    scaling = merge_csv_glob(
        (root / "h4").glob("fold_*/seed_*/*/codebook_count_scaling.csv"),
        root / "tables" / "codebook_count_scaling.csv",
    )
    transfer = merge_csv_glob(
        (root / "h4").glob("fold_*/seed_*/*/codebook_transfer_matrix.csv"),
        root / "tables" / "codebook_transfer_matrix.csv",
    )
    atomic_json(
        root / "h4" / "h4.done.json",
        {"completed": utc_now(), "config_hash": cfg_hash, "units": expected, "scaling_rows": scaling, "transfer_rows": transfer},
    )
    print(f"[merge-h4] units={expected} scaling={scaling} transfer={transfer}", flush=True)


def load_evidence_helpers() -> Any:
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_wan5b_nonorm_evidence as evidence

    return evidence


def h6_tasks(cfg: dict[str, Any]) -> list[tuple[int, Path]]:
    if not bool(cfg.get("h6", {}).get("enabled", False)):
        return []
    return [
        (index, resolve_path(path))
        for index, path in enumerate(cfg["h6"].get("capture_files", []))
    ]


def h6_task_paths(root: Path, index: int) -> dict[str, Path]:
    base = root / "h6" / f"capture_{index:02d}"
    return {
        "metrics": base / "functional_error_correlation.csv",
        "validation": base / "validation.json",
        "done": base / "done.json",
    }


def relative_mse(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.mean((candidate.float() - reference.float()) ** 2)
    denominator = torch.mean(reference.float() ** 2).clamp_min(1e-12)
    return float((numerator / denominator).item())


def max_abs_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.max(torch.abs(candidate.float() - reference.float())).item())


def apply_output_matrix(
    evidence: Any, output: torch.Tensor, matrix: torch.Tensor | None
) -> torch.Tensor:
    if matrix is None:
        return output
    return evidence.apply_camera_prope_matrix(output, matrix)


def head_attention_metrics(
    *,
    query: torch.Tensor,
    key_reference: torch.Tensor,
    key_candidate: torch.Tensor,
    value_reference: torch.Tensor,
    value_candidate: torch.Tensor,
    history_tokens: int,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    query_batch_size: int,
    topk: int,
    output_matrix: torch.Tensor | None,
    evidence: Any,
) -> dict[str, float]:
    """Compare one head using the same non-causal SDPA operation as the processor."""
    logit_reference_sq = 0.0
    logit_error_sq = 0.0
    output_reference_sq = 0.0
    output_error_sq = 0.0
    output_error_mse_sum = 0.0
    kl_sum = 0.0
    js_sum = 0.0
    overlap_sum = 0.0
    reference_history_mass_sum = 0.0
    candidate_history_mass_sum = 0.0
    age_shift_sum = 0.0
    query_count = 0
    scale = float(query.shape[-1]) ** -0.5
    key_count = int(key_reference.shape[-2])
    active_topk = min(int(topk), key_count)
    key_age_positions = key_positions.to(device=query.device, dtype=torch.float32)
    for start in range(0, int(query.shape[-2]), int(query_batch_size)):
        end = min(int(query.shape[-2]), start + int(query_batch_size))
        q = query[..., start:end, :]
        logits_reference = torch.matmul(
            q.float(), key_reference.float().transpose(-1, -2)
        ) * scale
        logits_candidate = torch.matmul(
            q.float(), key_candidate.float().transpose(-1, -2)
        ) * scale
        probs_reference = torch.softmax(logits_reference, dim=-1)
        probs_candidate = torch.softmax(logits_candidate, dim=-1)
        midpoint = ((probs_reference + probs_candidate) * 0.5).clamp_min(1e-20)
        p = probs_reference.clamp_min(1e-20)
        q_prob = probs_candidate.clamp_min(1e-20)
        rows = end - start
        query_count += rows
        logit_reference_sq += float(torch.sum(logits_reference**2).item())
        logit_error_sq += float(torch.sum((logits_candidate - logits_reference) ** 2).item())
        kl_sum += float(torch.sum(p * (torch.log(p) - torch.log(q_prob))).item())
        js_sum += 0.5 * float(
            torch.sum(p * (torch.log(p) - torch.log(midpoint))).item()
            + torch.sum(q_prob * (torch.log(q_prob) - torch.log(midpoint))).item()
        )
        reference_top = torch.topk(probs_reference, active_topk, dim=-1).indices
        candidate_top = torch.topk(probs_candidate, active_topk, dim=-1).indices
        overlap = (
            reference_top.unsqueeze(-1) == candidate_top.unsqueeze(-2)
        ).any(dim=-1).float().sum(dim=-1) / active_topk
        overlap_sum += float(overlap.sum().item())

        reference_history = probs_reference[..., :history_tokens]
        candidate_history = probs_candidate[..., :history_tokens]
        reference_mass = reference_history.sum(dim=-1)
        candidate_mass = candidate_history.sum(dim=-1)
        reference_history_mass_sum += float(reference_mass.sum().item())
        candidate_history_mass_sum += float(candidate_mass.sum().item())
        query_pos = query_positions[start:end].to(device=query.device, dtype=torch.float32)
        signed_age = query_pos[:, None] - key_age_positions[:history_tokens][None, :]
        reference_center = (
            reference_history.squeeze(0) * signed_age
        ).sum(dim=-1) / reference_mass.squeeze(0).clamp_min(1e-20)
        candidate_center = (
            candidate_history.squeeze(0) * signed_age
        ).sum(dim=-1) / candidate_mass.squeeze(0).clamp_min(1e-20)
        age_shift_sum += float((candidate_center - reference_center).sum().item())

        reference_output = F.scaled_dot_product_attention(
            q, key_reference, value_reference, dropout_p=0.0
        )
        candidate_output = F.scaled_dot_product_attention(
            q, key_candidate, value_candidate, dropout_p=0.0
        )
        matrix_slice = None if output_matrix is None else output_matrix[:, start:end]
        reference_output = apply_output_matrix(evidence, reference_output, matrix_slice)
        candidate_output = apply_output_matrix(evidence, candidate_output, matrix_slice)
        output_reference_sq += float(torch.sum(reference_output.float() ** 2).item())
        output_error_sq += float(
            torch.sum((candidate_output.float() - reference_output.float()) ** 2).item()
        )
        output_error_mse_sum += float(
            torch.mean((candidate_output.float() - reference_output.float()) ** 2).item()
        ) * rows

    return {
        "attention_logit_relative_l2": math.sqrt(logit_error_sq / max(logit_reference_sq, 1e-20)),
        "attention_kl_reference_to_candidate": kl_sum / query_count,
        "attention_js": js_sum / query_count,
        "attention_topk_overlap": overlap_sum / query_count,
        "attention_output_relative_l2": math.sqrt(output_error_sq / max(output_reference_sq, 1e-20)),
        "attention_output_mse": output_error_mse_sum / query_count,
        "reference_historical_attention_mass": reference_history_mass_sum / query_count,
        "candidate_historical_attention_mass": candidate_history_mass_sum / query_count,
        "historical_attention_mass_delta": (
            candidate_history_mass_sum - reference_history_mass_sum
        ) / query_count,
        "historical_signed_age_center_shift_tokens": age_shift_sum / query_count,
    }


def run_h6_task(
    cfg: dict[str, Any], cfg_hash: str, index: int, capture_path: Path, device: torch.device
) -> None:
    paths = h6_task_paths(output_root(cfg), index)
    if paths["done"].exists():
        return
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite incomplete H6 outputs: {existing}")
    if not capture_path.exists():
        raise FileNotFoundError(capture_path)
    started = time.time()
    evidence = load_evidence_helpers()
    raw = torch.load(capture_path, map_location="cpu", weights_only=False)
    metadata = dict(raw.get("metadata", {}))
    required = {
        "query_pre", "key_pre", "value_pre", "query_post_rope", "key_post_rope",
        "value_post_rope", "query_post_prope", "key_post_prope", "value_post_prope",
        "query_rope_cos", "query_rope_sin", "key_rope_cos", "key_rope_sin",
        "query_viewmats", "query_Ks", "key_viewmats", "key_Ks",
        "query_position_indices", "key_position_indices", "key_indices",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"H6 capture is missing exact-replay fields: {missing}")
    if metadata.get("alignment") != "exact_same_call_same_indices":
        raise ValueError(f"H6 capture is not exactly aligned: {metadata}")
    layer = int(metadata["layer"])
    history_tokens = int(metadata["cache_tokens"])
    key_tokens = int(raw["key_pre"].shape[2])
    if int(metadata["key_sample"]) != int(metadata["key_len"]) or key_tokens != int(metadata["key_len"]):
        raise ValueError("H6 requires a complete, not subsampled, key axis")
    if not torch.equal(raw["key_indices"], torch.arange(key_tokens)):
        raise ValueError("H6 key samples do not preserve the complete canonical key order")
    tensor_keys = [
        "query_pre", "key_pre", "value_pre", "query_post_rope", "key_post_rope",
        "value_post_rope", "query_post_prope", "key_post_prope", "value_post_prope",
        "query_rope_cos", "query_rope_sin", "key_rope_cos", "key_rope_sin",
        "query_viewmats", "query_Ks", "key_viewmats", "key_Ks",
    ]
    tensors = {key: raw[key].to(device=device).contiguous() for key in tensor_keys}
    query_positions = raw["query_position_indices"].long().clone()
    key_positions = raw["key_position_indices"].long().clone()

    run_info_path = capture_path.parents[3] / "run_info.json"
    parent_run_info = {}
    if run_info_path.exists():
        with run_info_path.open("r", encoding="utf-8") as handle:
            parent_run_info = json.load(handle)
    prompt_list = parent_run_info.get("prompt_list", ["unknown_capture_prompt"])
    prompt_text = str(prompt_list[0]) if prompt_list else "unknown_capture_prompt"
    del raw

    codebook_root = Path(cfg["h6"]["codebook_root"])
    k_codebooks = codebook_root / cfg["h6"]["k_codebook_dir"]
    v_codebooks = codebook_root / cfg["h6"]["v_codebook_dir"]
    evidence.validate_codebook_dir(k_codebooks, layer=layer, heads=HEADS, expected_dim=DIM)
    evidence.validate_codebook_dir(v_codebooks, layer=layer, heads=HEADS, expected_dim=DIM)
    packed_k = evidence.encode_fixed(
        tensors["key_pre"][:, :, :history_tokens], layer=layer, directory=k_codebooks,
        batch_size=int(cfg["rvq"]["assignment_batch_size"]),
    )
    reconstructed_k_history = evidence.rvq_dequantize_tensor_cache(
        packed_k, tensors["key_pre"].dtype
    )
    del packed_k
    packed_v = evidence.encode_fixed(
        tensors["value_pre"][:, :, :history_tokens], layer=layer, directory=v_codebooks,
        batch_size=int(cfg["rvq"]["assignment_batch_size"]),
    )
    reconstructed_v_history = evidence.rvq_dequantize_tensor_cache(
        packed_v, tensors["value_pre"].dtype
    )
    del packed_v

    standard_k_history = evidence.apply_saved_rope(
        reconstructed_k_history, tensors["key_rope_cos"][:, :, :history_tokens],
        tensors["key_rope_sin"][:, :, :history_tokens],
    )
    standard_v_history = reconstructed_v_history
    reconstructed_standard_k = torch.cat(
        [standard_k_history, tensors["key_post_rope"][:, :, history_tokens:]], dim=2
    )
    reconstructed_standard_v = torch.cat(
        [standard_v_history, tensors["value_post_rope"][:, :, history_tokens:]], dim=2
    )
    query_q_matrix, _query_kv_matrix, query_output_matrix = evidence.camera_prope_matrices(
        tensors["query_viewmats"], tensors["query_Ks"]
    )
    _history_q_matrix, history_kv_matrix, _history_output = evidence.camera_prope_matrices(
        tensors["key_viewmats"][:, :history_tokens], tensors["key_Ks"][:, :history_tokens]
    )
    prope_k_history = evidence.apply_camera_prope_matrix(
        reconstructed_k_history, history_kv_matrix
    )
    prope_v_history = evidence.apply_camera_prope_matrix(
        reconstructed_v_history, history_kv_matrix
    )
    reconstructed_prope_k = torch.cat(
        [prope_k_history, tensors["key_post_prope"][:, :, history_tokens:]], dim=2
    )
    reconstructed_prope_v = torch.cat(
        [prope_v_history, tensors["value_post_prope"][:, :, history_tokens:]], dim=2
    )

    # Exact input-transform gates: failure invalidates all functional rows.
    replay_standard_q = evidence.apply_saved_rope(
        tensors["query_pre"], tensors["query_rope_cos"], tensors["query_rope_sin"]
    )
    replay_standard_k = evidence.apply_saved_rope(
        tensors["key_pre"], tensors["key_rope_cos"], tensors["key_rope_sin"]
    )
    replay_prope_q = evidence.apply_camera_prope_matrix(tensors["query_pre"], query_q_matrix)
    _full_q_matrix, full_kv_matrix, _full_output = evidence.camera_prope_matrices(
        tensors["key_viewmats"], tensors["key_Ks"]
    )
    replay_prope_k = evidence.apply_camera_prope_matrix(tensors["key_pre"], full_kv_matrix)
    replay_prope_v = evidence.apply_camera_prope_matrix(tensors["value_pre"], full_kv_matrix)
    validation = {
        "capture_path": str(capture_path),
        "capture_sha256": sha256_file(capture_path),
        "alignment": metadata.get("alignment"),
        "layer": layer,
        "history_tokens": history_tokens,
        "current_tokens": key_tokens - history_tokens,
        "key_axis_complete_and_canonical": True,
        "processor_attention_semantics": "two non-causal SDPA halves: standard RoPE and camera PRoPE",
        "functional_output_scope": "SDPA output; camera branch includes exact apply_fn_o equivalent; learned to_out layers unavailable",
        "standard_query_replay_max_abs": max_abs_error(replay_standard_q, tensors["query_post_rope"]),
        "standard_key_replay_max_abs": max_abs_error(replay_standard_k, tensors["key_post_rope"]),
        "standard_value_replay_max_abs": max_abs_error(tensors["value_pre"], tensors["value_post_rope"]),
        "prope_query_replay_max_abs": max_abs_error(replay_prope_q, tensors["query_post_prope"]),
        "prope_key_replay_max_abs": max_abs_error(replay_prope_k, tensors["key_post_prope"]),
        "prope_value_replay_max_abs": max_abs_error(replay_prope_v, tensors["value_post_prope"]),
        "current_standard_k_max_abs": max_abs_error(
            reconstructed_standard_k[:, :, history_tokens:], tensors["key_post_rope"][:, :, history_tokens:]
        ),
        "current_standard_v_max_abs": max_abs_error(
            reconstructed_standard_v[:, :, history_tokens:], tensors["value_post_rope"][:, :, history_tokens:]
        ),
        "current_prope_k_max_abs": max_abs_error(
            reconstructed_prope_k[:, :, history_tokens:], tensors["key_post_prope"][:, :, history_tokens:]
        ),
        "current_prope_v_max_abs": max_abs_error(
            reconstructed_prope_v[:, :, history_tokens:], tensors["value_post_prope"][:, :, history_tokens:]
        ),
    }
    replay_error_keys = [key for key in validation if key.endswith("_replay_max_abs")]
    validation["exact_transform_replay"] = all(float(validation[key]) == 0.0 for key in replay_error_keys)
    validation["current_segment_untouched"] = all(
        float(validation[key]) == 0.0
        for key in validation
        if key.startswith("current_") and key.endswith("_max_abs")
    )
    if not validation["exact_transform_replay"] or not validation["current_segment_untouched"]:
        raise RuntimeError(f"H6 exact-replay gate failed: {validation}")

    rows: list[dict[str, Any]] = []
    halves = {
        "standard_rope": (
            tensors["query_post_rope"], tensors["key_post_rope"], tensors["value_post_rope"],
            reconstructed_standard_k, reconstructed_standard_v, None,
        ),
        "camera_prope": (
            tensors["query_post_prope"], tensors["key_post_prope"], tensors["value_post_prope"],
            reconstructed_prope_k, reconstructed_prope_v, query_output_matrix,
        ),
    }
    k_rel_mse = [
        relative_mse(reconstructed_k_history[:, head], tensors["key_pre"][:, head, :history_tokens])
        for head in range(HEADS)
    ]
    v_rel_mse = [
        relative_mse(reconstructed_v_history[:, head], tensors["value_pre"][:, head, :history_tokens])
        for head in range(HEADS)
    ]
    for half_name, (query, key_ref, value_ref, key_hat, value_hat, output_matrix) in halves.items():
        for head in range(HEADS):
            head_matrix = output_matrix
            for intervention, candidate_key, candidate_value in (
                ("K_only", key_hat[:, head : head + 1], value_ref[:, head : head + 1]),
                ("V_only", key_ref[:, head : head + 1], value_hat[:, head : head + 1]),
                ("K_and_V", key_hat[:, head : head + 1], value_hat[:, head : head + 1]),
            ):
                metrics = head_attention_metrics(
                    query=query[:, head : head + 1],
                    key_reference=key_ref[:, head : head + 1],
                    key_candidate=candidate_key,
                    value_reference=value_ref[:, head : head + 1],
                    value_candidate=candidate_value,
                    history_tokens=history_tokens,
                    query_positions=query_positions,
                    key_positions=key_positions,
                    query_batch_size=int(cfg["h6"]["query_batch_size"]),
                    topk=int(cfg["h6"]["topk"]),
                    output_matrix=head_matrix,
                    evidence=evidence,
                )
                rows.append(
                    {
                        "study_id": cfg["study_id"], "config_hash": cfg_hash,
                        "hypothesis_id": "H6", "prompt_id": 0, "prompt": prompt_text,
                        "capture_index": index, "capture_path": str(capture_path),
                        "layer": layer, "head": head, "K_or_V": intervention,
                        "hook_stage": "pre_positional_historical_only",
                        "attention_half": half_name, "intervention": intervention,
                        "history_tokens": history_tokens,
                        "current_bf16_tokens": key_tokens - history_tokens,
                        "rvq_levels": 8, "codebook_size": 64,
                        "runtime_codebook_storage": "int4", "input_norm_mode": "none",
                        "historical_K_relative_mse": k_rel_mse[head],
                        "historical_V_relative_mse": v_rel_mse[head],
                        **metrics,
                    }
                )
    write_csv(paths["metrics"], rows)
    atomic_json(paths["validation"], validation)
    atomic_json(
        paths["done"],
        {
            "completed": utc_now(), "config_hash": cfg_hash, "capture_index": index,
            "layer": layer, "rows": len(rows), "seconds": time.time() - started,
        },
    )
    print(f"[h6] capture={index} layer={layer} rows={len(rows)} seconds={time.time()-started:.1f}", flush=True)


def cmd_h6_worker(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    if not torch.cuda.is_available():
        raise RuntimeError("H6 worker requires CUDA")
    tasks = h6_tasks(cfg)
    selected = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_id]
    print(f"[h6-worker] shard={args.shard_id}/{args.num_shards} tasks={len(selected)}", flush=True)
    for task in selected:
        run_h6_task(cfg, cfg_hash, *task, torch.device("cuda:0"))


def cmd_merge_h6(args: argparse.Namespace) -> None:
    cfg_path = resolve_path(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    root = output_root(cfg)
    expected = len(h6_tasks(cfg))
    done = list((root / "h6").glob("capture_*/done.json"))
    if len(done) != expected:
        raise RuntimeError(f"H6 incomplete: expected {expected}, found {len(done)}")
    rows = merge_csv_glob(
        (root / "h6").glob("capture_*/functional_error_correlation.csv"),
        root / "tables" / "functional_error_correlation.csv",
    )
    atomic_json(
        root / "h6" / "h6.done.json",
        {"completed": utc_now(), "config_hash": cfg_hash, "units": expected, "rows": rows},
    )
    print(f"[merge-h6] units={expected} rows={rows}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare", "merge-cv", "merge-secondary", "merge-h2-sweep", "merge-h4", "merge-h6"
    ):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
    for name in ("cv", "secondary", "h2-sweep", "h4", "h6"):
        launch = sub.add_parser(name)
        launch.add_argument("--config", required=True)
        launch.add_argument("--gpus", default="")
    for name in ("cv-worker", "secondary-worker", "h2-sweep-worker", "h4-worker", "h6-worker"):
        worker = sub.add_parser(name)
        worker.add_argument("--config", required=True)
        worker.add_argument("--shard-id", type=int, required=True)
        worker.add_argument("--num-shards", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "cv":
        launch_workers(args, "cv-worker")
    elif args.command == "cv-worker":
        cmd_cv_worker(args)
    elif args.command == "merge-cv":
        cmd_merge_cv(args)
    elif args.command == "secondary":
        launch_workers(args, "secondary-worker")
    elif args.command == "secondary-worker":
        cmd_secondary_worker(args)
    elif args.command == "merge-secondary":
        cmd_merge_secondary(args)
    elif args.command == "h2-sweep":
        launch_workers(args, "h2-sweep-worker")
    elif args.command == "h2-sweep-worker":
        cmd_h2_sweep_worker(args)
    elif args.command == "merge-h2-sweep":
        cmd_merge_h2_sweep(args)
    elif args.command == "h4":
        launch_workers(args, "h4-worker")
    elif args.command == "h4-worker":
        cmd_h4_worker(args)
    elif args.command == "merge-h4":
        cmd_merge_h4(args)
    elif args.command == "h6":
        launch_workers(args, "h6-worker")
    elif args.command == "h6-worker":
        cmd_h6_worker(args)
    elif args.command == "merge-h6":
        cmd_merge_h6(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
