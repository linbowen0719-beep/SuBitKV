#!/usr/bin/env python3
"""Train and assemble attention-functional variable-cardinality RVQ codebooks.

The experiment keeps the validated 30-bit uniform-L5/compact30 schedules and
changes one factor at a time:

* Q-aware K refinement uses the empirical per-layer/head query covariance.
* V-functional refinement minimizes sampled attention-output error and an
  explicit positive cross-token coherence term.
* K/V-asymmetric allocation changes only which tensor/layer uses uniform or
  compact codebooks, with ten uniform tensor-layer assignments in both maps.

Initial codebooks are the actual INT4-dequantized codebooks used by the P1
all-layer exact replay.  Three initialization seeds are preserved.  Runtime
directories use one list-valued per-head payload so the real cache path can
pack every 30-bit schedule into one int32 index container.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_p1_compact as compact  # noqa: E402
import analyze_wan5b_p1_exact_replay as exact  # noqa: E402
import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402


VARIANTS = {
    "fullmap_baseline": {"k_kind": "base", "v_kind": "base", "policy": "fullmap"},
    "qaware_k": {"k_kind": "refined", "v_kind": "base", "policy": "fullmap"},
    "vaware_v": {"k_kind": "base", "v_kind": "refined", "policy": "fullmap"},
    "qaware_vaware": {"k_kind": "refined", "v_kind": "refined", "policy": "fullmap"},
    "kv_independent": {"k_kind": "base", "v_kind": "base", "policy": "independent"},
    "combined": {"k_kind": "refined", "v_kind": "refined", "policy": "independent"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "worker", "launch", "assemble", "summarize", "all"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, required=True)
        if command == "worker":
            sub.add_argument("--shard-id", type=int, required=True)
            sub.add_argument("--num-shards", type=int, required=True)
        if command in {"launch", "all"}:
            sub.add_argument("--gpus", default="")
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], str, Path]:
    resolved = p0.resolve_path(path)
    cfg, cfg_hash = p0.parse_config(resolved)
    return cfg, cfg_hash, resolved


def output_root(cfg: dict[str, Any]) -> Path:
    return p0.resolve_path(cfg["output_dir"])


def layers(cfg: dict[str, Any]) -> list[int]:
    return p0.parse_axis(cfg.get("layers", "all"), p0.LAYERS)


def heads(cfg: dict[str, Any]) -> list[int]:
    return p0.parse_axis(cfg.get("heads", "all"), p0.HEADS)


def source_codebook_path(cfg: dict[str, Any], layer: int) -> Path:
    return (
        p0.resolve_path(cfg["source_exact_root"])
        / "exact_replay"
        / f"capture_{int(layer):02d}"
        / "calibrated_int4_codebooks.pt"
    )


def dequantize_packed_level(payload: dict[str, Any]) -> torch.Tensor:
    if str(payload.get("storage_dtype", "")).lower() != "int4":
        raise ValueError(f"Expected INT4 source codebook, got {payload.get('storage_dtype')}")
    q = exact.unpack_uint4(payload["q_packed"], payload["q_shape"]).to(torch.int16) - 8
    return (q.float() * payload["scale"].float()).contiguous()


def load_initial_codebooks(
    cfg: dict[str, Any], layer: int, seed: int, tensor_name: str, method: str
) -> list[torch.Tensor]:
    path = source_codebook_path(cfg, layer)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["layer"]) != int(layer):
        raise ValueError(f"Source codebook layer mismatch: {path}")
    packed = payload["packed_codebooks"][int(seed)][str(tensor_name)][str(method)]
    result = [dequantize_packed_level(value) for value in packed]
    expected = [int(value) for value in cfg["methods"][method]]
    if [int(value.shape[1]) for value in result] != expected:
        raise ValueError(f"Source schedule mismatch layer={layer} seed={seed} {tensor_name} {method}")
    return result


def collect_samples(
    cfg: dict[str, Any], tensor_name: str, layer: int, prompt_ids: Sequence[int], part: str
) -> torch.Tensor:
    local = dict(cfg)
    local["layers"] = [int(layer)]
    local["heads"] = heads(cfg)
    return compact.collect_samples(local, tensor_name, layer, prompt_ids, part).float()


def fake_int4(codebook: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return codebook
    scale = (codebook.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0)
    quantized = torch.round(codebook / scale).clamp(-7, 7) * scale
    return codebook + (quantized - codebook).detach()


@torch.no_grad()
def nearest_indices(residual: torch.Tensor, centroids: torch.Tensor, batch_size: int) -> torch.Tensor:
    chunks = []
    centroid_sq = (centroids.float() * centroids.float()).sum(dim=-1)
    for start in range(0, int(residual.shape[1]), int(batch_size)):
        end = min(int(residual.shape[1]), start + int(batch_size))
        value = residual[:, start:end].float()
        distance = (
            (value * value).sum(dim=-1, keepdim=True)
            + centroid_sq[:, None]
            - 2.0 * torch.bmm(value, centroids.float().transpose(1, 2))
        )
        chunks.append(torch.argmin(distance, dim=-1))
    return torch.cat(chunks, dim=1)


def reconstruct_with_codebooks(
    value: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    *,
    assignment_batch_size: int,
    int4_qat: bool,
) -> torch.Tensor:
    reconstruction = torch.zeros_like(value, dtype=torch.float32)
    residual = value.float()
    for parameter in parameters:
        centroids = fake_int4(parameter.float(), int4_qat)
        ids = nearest_indices(residual.detach(), centroids.detach(), assignment_batch_size)
        assigned = torch.gather(
            centroids,
            1,
            ids[..., None].expand(-1, -1, int(value.shape[-1])),
        )
        reconstruction = reconstruction + assigned
        residual = value.float() - reconstruction
    return reconstruction


def relative_mse(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (candidate.float() - reference.float()).square().mean() / reference.float().square().mean().clamp_min(1e-12)


def query_covariance(query: torch.Tensor) -> torch.Tensor:
    covariance = torch.bmm(query.float().transpose(1, 2), query.float()) / max(1, int(query.shape[1]))
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return covariance * (float(query.shape[-1]) / trace[..., None])


def q_metric_error(error: torch.Tensor, covariance: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    projected = torch.bmm(error.float(), covariance.float())
    numerator = (projected * error.float()).sum(dim=-1).mean()
    reference_projected = torch.bmm(reference.float(), covariance.float())
    denominator = (reference_projected * reference.float()).sum(dim=-1).mean().clamp_min(1e-12)
    return numerator / denominator


def v_functional_terms(
    query: torch.Tensor,
    key: torch.Tensor,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    q_samples: int,
    key_samples: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_take = min(int(q_samples), int(query.shape[1]))
    kv_take = min(int(key_samples), int(key.shape[1]))
    q_indices = torch.randperm(int(query.shape[1]), generator=generator, device=query.device)[:q_take]
    kv_indices = torch.randperm(int(key.shape[1]), generator=generator, device=key.device)[:kv_take]
    q = query.index_select(1, q_indices).float()
    k = key.index_select(1, kv_indices).float()
    v = reference.index_select(1, kv_indices).float()
    v_hat = candidate.index_select(1, kv_indices).float()
    attention = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(float(k.shape[-1])), dim=-1)
    output = torch.bmm(attention, v)
    output_hat = torch.bmm(attention, v_hat)
    output_loss = relative_mse(output_hat, output)

    error = v_hat - v
    weights = attention.mean(dim=1)
    summed = (weights[..., None] * error).sum(dim=1)
    total_energy = summed.square().sum(dim=-1)
    diagonal_energy = (weights.square()[..., None] * error.square()).sum(dim=(1, 2))
    cross_energy = (total_energy - diagonal_energy).clamp_min(0.0)
    reference_energy = output.square().sum(dim=-1).mean(dim=-1).clamp_min(1e-12)
    coherence = (cross_energy / reference_energy).mean()
    return output_loss, coherence


@torch.no_grad()
def evaluate_codebooks(
    tensor_name: str,
    codebooks: Sequence[torch.Tensor],
    data: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    training = cfg["training"]
    value = data[tensor_name].to(device=device, dtype=torch.float32)
    runtime = [fake_int4(item.to(device=device, dtype=torch.float32), bool(training["int4_qat"])) for item in codebooks]
    reconstruction = reconstruct_with_codebooks(
        value,
        runtime,
        assignment_batch_size=int(training["batch_size"]),
        int4_qat=False,
    )
    result = {"relative_mse": float(relative_mse(reconstruction, value).item())}
    if tensor_name == "K":
        q = data["Q"].to(device=device, dtype=torch.float32)
        covariance = query_covariance(q)
        result["q_metric_relative_error"] = float(
            q_metric_error(reconstruction - value, covariance, value).item()
        )
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed) + 700_001)
        output, coherence = v_functional_terms(
            data["Q"].to(device=device, dtype=torch.float32),
            data["K"].to(device=device, dtype=torch.float32),
            value,
            reconstruction,
            q_samples=int(training["q_samples"]),
            key_samples=int(training["v_key_samples"]),
            generator=generator,
        )
        result["v_output_relative_error"] = float(output.item())
        result["v_cross_token_coherence"] = float(coherence.item())
    return result


def refine_codebooks(
    tensor_name: str,
    initial: Sequence[torch.Tensor],
    train_data: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[list[torch.Tensor], list[dict[str, float]]]:
    training = cfg["training"]
    parameters = [
        torch.nn.Parameter(value.to(device=device, dtype=torch.float32).clone())
        for value in initial
    ]
    originals = [value.to(device=device, dtype=torch.float32) for value in initial]
    optimizer = torch.optim.Adam(parameters, lr=float(training["learning_rate"]))
    value_cpu = train_data[tensor_name]
    query_all = train_data["Q"].to(device=device, dtype=torch.float32)
    key_all = train_data["K"].to(device=device, dtype=torch.float32)
    covariance = query_covariance(query_all) if tensor_name == "K" else None
    generator = torch.Generator(device=device).manual_seed(int(seed) + (0 if tensor_name == "K" else 900_001))
    history = []
    batch_size = int(training["batch_size"])
    for epoch in range(int(training["epochs"])):
        permutation = torch.randperm(
            int(value_cpu.shape[1]),
            generator=torch.Generator(device="cpu").manual_seed(int(seed) + epoch * 100_003),
        )
        sums = {"loss": 0.0, "rec": 0.0, "functional": 0.0, "coherence": 0.0, "steps": 0.0}
        for start in range(0, int(permutation.numel()), batch_size):
            ids = permutation[start : start + batch_size]
            value = value_cpu.index_select(1, ids).to(device=device, dtype=torch.float32)
            reconstruction = reconstruct_with_codebooks(
                value,
                parameters,
                assignment_batch_size=batch_size,
                int4_qat=bool(training["int4_qat"]),
            )
            rec = relative_mse(reconstruction, value)
            coherence = torch.zeros((), device=device)
            if tensor_name == "K":
                assert covariance is not None
                functional = q_metric_error(reconstruction - value, covariance, value)
            else:
                functional, coherence = v_functional_terms(
                    query_all,
                    key_all.index_select(1, ids.to(device)),
                    value,
                    reconstruction,
                    q_samples=int(training["q_samples"]),
                    key_samples=min(int(training["v_key_samples"]), int(value.shape[1])),
                    generator=generator,
                )
            drift = torch.stack(
                [relative_mse(parameter, original) for parameter, original in zip(parameters, originals)]
            ).mean()
            loss = (
                functional
                + float(training["reconstruction_weight"]) * rec
                + float(training["drift_weight"]) * drift
                + (float(training["v_coherence_weight"]) * coherence if tensor_name == "V" else 0.0)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip"]))
            optimizer.step()
            sums["loss"] += float(loss.detach().item())
            sums["rec"] += float(rec.detach().item())
            sums["functional"] += float(functional.detach().item())
            sums["coherence"] += float(coherence.detach().item())
            sums["steps"] += 1.0
        history.append(
            {
                "epoch": float(epoch),
                **{key: value / max(1.0, sums["steps"]) for key, value in sums.items() if key != "steps"},
            }
        )
    return [value.detach().cpu().float().contiguous() for value in parameters], history


def bank_dir(cfg: dict[str, Any], seed: int, kind: str, tensor_name: str, method: str) -> Path:
    return output_root(cfg) / "banks" / f"seed_{int(seed)}" / kind / tensor_name.lower() / method


def save_codebooks(
    cfg: dict[str, Any], cfg_hash: str, seed: int, kind: str, tensor_name: str,
    method: str, layer: int, codebooks: Sequence[torch.Tensor], objective: str,
) -> None:
    directory = bank_dir(cfg, seed, kind, tensor_name, method)
    directory.mkdir(parents=True, exist_ok=True)
    schedule = [int(value) for value in cfg["methods"][method]]
    selected_heads = heads(cfg)
    for local_head, head in enumerate(selected_heads):
        path = directory / f"layer_{int(layer)}_head_{int(head)}.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save(
            {
                "codebooks": [value[local_head].to(torch.bfloat16).contiguous() for value in codebooks],
                "input_norm_mode": "none",
                "rvq_format": "variable_cardinality",
                "config": {
                    "study_id": cfg["study_id"],
                    "config_hash": cfg_hash,
                    "initialization_seed": int(seed),
                    "tensor": tensor_name,
                    "layer": int(layer),
                    "head": int(head),
                    "kind": kind,
                    "objective": objective,
                    "schedule": schedule,
                    "logical_index_bits": compact.schedule_bits(schedule),
                    "index_container": "int32",
                    "input_norm_mode": "none",
                    "source": str(source_codebook_path(cfg, layer)),
                },
            },
            path,
        )


def task_done_path(cfg: dict[str, Any], layer: int, seed: int) -> Path:
    return output_root(cfg) / "status" / "training" / f"layer_{int(layer):02d}_seed_{int(seed)}.done.json"


def train_task(cfg: dict[str, Any], cfg_hash: str, layer: int, seed: int, device: torch.device) -> None:
    done = task_done_path(cfg, layer, seed)
    if done.exists():
        return
    partials = list((output_root(cfg) / "banks" / f"seed_{seed}").glob(f"**/layer_{layer}_head_*.pt"))
    if partials:
        raise FileExistsError(f"Refusing to overwrite incomplete task layer={layer} seed={seed}: {partials[:3]}")
    prompt_ids = list(range(20))
    train_data = {
        name: collect_samples(cfg, name, layer, prompt_ids, "train")
        for name in ("Q", "K", "V")
    }
    eval_data = {
        name: collect_samples(cfg, name, layer, prompt_ids, "eval")
        for name in ("Q", "K", "V")
    }
    records = []
    started = time.time()
    for tensor_name in ("K", "V"):
        for method in cfg["methods"]:
            initial = load_initial_codebooks(cfg, layer, seed, tensor_name, method)
            base_metrics = evaluate_codebooks(
                tensor_name, initial, eval_data, cfg, device,
                seed + layer * 10_000 + (0 if tensor_name == "K" else 500_000),
            )
            refined, history = refine_codebooks(
                tensor_name, initial, train_data, cfg, device,
                seed + layer * 10_000 + (0 if tensor_name == "K" else 500_000),
            )
            refined_metrics = evaluate_codebooks(
                tensor_name, refined, eval_data, cfg, device,
                seed + layer * 10_000 + (0 if tensor_name == "K" else 500_000),
            )
            save_codebooks(
                cfg, cfg_hash, seed, "base", tensor_name, method, layer, initial,
                "euclidean_kmeans_int4_source",
            )
            save_codebooks(
                cfg, cfg_hash, seed, "refined", tensor_name, method, layer, refined,
                "query_covariance" if tensor_name == "K" else "attention_output_plus_coherence",
            )
            records.append(
                {
                    "layer": int(layer),
                    "initialization_seed": int(seed),
                    "tensor": tensor_name,
                    "method": method,
                    "schedule": [int(value) for value in cfg["methods"][method]],
                    "base": base_metrics,
                    "refined": refined_metrics,
                    "relative_changes": {
                        key: (float(refined_metrics[key]) - float(base_metrics[key]))
                        / max(abs(float(base_metrics[key])), 1e-12)
                        for key in base_metrics
                    },
                    "training_history": history,
                }
            )
            torch.cuda.empty_cache()
    metric_path = output_root(cfg) / "training" / f"layer_{layer:02d}_seed_{seed}.json"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    p0.atomic_json(metric_path, {"config_hash": cfg_hash, "records": records})
    done.parent.mkdir(parents=True, exist_ok=True)
    p0.atomic_json(
        done,
        {
            "completed": p0.utc_now(),
            "config_hash": cfg_hash,
            "layer": int(layer),
            "initialization_seed": int(seed),
            "seconds": time.time() - started,
            "metrics": str(metric_path),
        },
    )
    print(f"[trained] layer={layer} seed={seed} seconds={time.time()-started:.1f}", flush=True)


def prepare(cfg: dict[str, Any], cfg_hash: str, cfg_path: Path) -> None:
    root = output_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    run_info_path = root / "run_info.json"
    if run_info_path.exists():
        existing = json.loads(run_info_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != cfg_hash:
            raise FileExistsError(f"Output root belongs to a different config: {root}")
        print(f"[prepare-skip] {run_info_path}")
        return
    registry = p0.resolve_path(cfg["hypothesis_registry"])
    source_manifest = p0.resolve_path(cfg["source_exact_root"]) / "exact_replay" / "codebook_manifest.json"
    prompt_items = p0.prompt_items(Path(cfg["data_root"]))
    p0.atomic_json(
        run_info_path,
        {
            "git_commit": p0.git_commit(),
            "config_path": str(cfg_path.resolve()),
            "config_hash": cfg_hash,
            "codebook_path": {"source_exact": str(source_manifest.resolve()), "runtime": None},
            "codebook_hash": {"source_exact": p0.sha256_file(source_manifest), "runtime": None},
            "hostname": p0.socket.gethostname(),
            "gpu_type": p0.gpu_name(),
            "seed": int(cfg["seed"]),
            "initialization_seeds": [int(value) for value in cfg["initialization_seeds"]],
            "prompt_list": [str(item["prompt"]) for item in prompt_items],
            "timestamp": p0.utc_now(),
            "model_id": cfg["model_id"],
            "hypothesis_registry": str(registry.resolve()),
            "hypothesis_registry_hash": p0.sha256_file(registry),
            "source_data": str(Path(cfg["data_root"]) / "kv_dump" / "train"),
            "source_data_read_only": True,
            "statistical_unit": "prompt",
            "index_rate": "30 logical bits in one int32 container for every schedule",
        },
    )
    shutil.copy2(cfg_path, root / "config.yaml")
    print(f"[prepare] root={root} config_hash={cfg_hash}", flush=True)


def worker(cfg: dict[str, Any], cfg_hash: str, shard_id: int, num_shards: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for functional RVQ refinement")
    device = torch.device("cuda:0")
    tasks = [
        (layer, int(seed))
        for layer in layers(cfg)
        for seed in cfg["initialization_seeds"]
    ]
    selected = [task for index, task in enumerate(tasks) if index % int(num_shards) == int(shard_id)]
    print(f"[worker] shard={shard_id}/{num_shards} tasks={len(selected)} gpu={torch.cuda.get_device_name(0)}")
    for layer, seed in selected:
        train_task(cfg, cfg_hash, layer, seed, device)


def launch(cfg: dict[str, Any], cfg_path: Path, raw_gpus: str) -> None:
    gpu_ids = [int(value) for value in (raw_gpus.split(",") if raw_gpus else cfg["gpus"])]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError(f"Expected four distinct GPUs, got {gpu_ids}")
    log_dir = output_root(cfg) / "logs" / f"train_{p0.utc_now().replace(':', '').replace('+', '_')}"
    log_dir.mkdir(parents=True, exist_ok=False)
    processes = []
    for shard, gpu in enumerate(gpu_ids):
        log_path = log_dir / f"gpu_{gpu}_shard_{shard}.log"
        handle = log_path.open("x", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        command = [
            str(cfg["python"]), str(Path(__file__).resolve()), "worker",
            "--config", str(cfg_path), "--shard-id", str(shard),
            "--num-shards", str(len(gpu_ids)),
        ]
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((gpu, process, handle, log_path))
        print(f"[launch] gpu={gpu} pid={process.pid} log={log_path}", flush=True)
    last_report = 0.0
    try:
        while any(process.poll() is None for _gpu, process, _handle, _path in processes):
            if time.time() - last_report >= 15:
                print(
                    "[monitor] " + ", ".join(
                        f"gpu{gpu}:pid{process.pid}:{'running' if process.poll() is None else process.returncode}"
                        for gpu, process, _handle, _path in processes
                    ),
                    flush=True,
                )
                last_report = time.time()
            time.sleep(2)
    finally:
        for _gpu, _process, handle, _path in processes:
            handle.close()
    failures = [item for item in processes if item[1].returncode != 0]
    if failures:
        for gpu, process, _handle, path in failures:
            print(f"[failed] gpu={gpu} rc={process.returncode} log={path}\n{path.read_text()[-12000:]}")
        raise RuntimeError(f"{len(failures)} training workers failed")


def method_for_layer(cfg: dict[str, Any], policy: str, tensor_name: str, layer: int) -> str:
    uniform = {
        int(value)
        for value in cfg["policies"][f"{policy}_uniform_{tensor_name}"]
    }
    return "uniform_l5_c64" if int(layer) in uniform else "compact30"


def assemble(cfg: dict[str, Any], cfg_hash: str) -> None:
    root = output_root(cfg)
    manifest_path = root / "runtime" / "manifest.json"
    if manifest_path.exists():
        print(f"[assemble-skip] {manifest_path}")
        return
    files = []
    for seed in cfg["initialization_seeds"]:
        for variant, spec in VARIANTS.items():
            for tensor_name in ("K", "V"):
                kind = spec[f"{tensor_name.lower()}_kind"]
                method_root = root / "runtime" / f"seed_{int(seed)}" / variant / tensor_name.lower()
                method_root.mkdir(parents=True, exist_ok=True)
                for layer in layers(cfg):
                    method = method_for_layer(cfg, str(spec["policy"]), tensor_name, layer)
                    source_dir = bank_dir(cfg, int(seed), str(kind), tensor_name, method)
                    for head in heads(cfg):
                        source = source_dir / f"layer_{layer}_head_{head}.pt"
                        destination = method_root / source.name
                        if not source.exists():
                            raise FileNotFoundError(source)
                        if destination.exists():
                            raise FileExistsError(destination)
                        shutil.copy2(source, destination)
                        files.append(
                            {
                                "path": str(destination.relative_to(root)),
                                "sha256": p0.sha256_file(destination),
                                "bytes": destination.stat().st_size,
                            }
                        )
    manifest = {
        "schema_version": "wan5b_functional_variable_rvq_runtime.v1",
        "config_hash": cfg_hash,
        "variants": VARIANTS,
        "files": files,
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "index_payload": "30 logical bits in one int32 per vector",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    p0.atomic_json(manifest_path, manifest)
    info_path = root / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["codebook_path"]["runtime"] = str(manifest_path.resolve())
    info["codebook_hash"]["runtime"] = p0.sha256_file(manifest_path)
    p0.atomic_json(info_path, info)
    print(f"[assemble] files={len(files)} bytes={manifest['total_bytes']} manifest={manifest_path}")


def summarize_training(cfg: dict[str, Any], cfg_hash: str) -> None:
    paths = sorted((output_root(cfg) / "training").glob("layer_*_seed_*.json"))
    expected = len(layers(cfg)) * len(cfg["initialization_seeds"])
    if len(paths) != expected:
        raise RuntimeError(f"Incomplete training metrics: {len(paths)}/{expected}")
    records = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config_hash") != cfg_hash:
            raise ValueError(f"Config hash mismatch: {path}")
        records.extend(payload["records"])
    grouped: dict[tuple[str, str, str], list[float]] = {}
    layer_grouped: dict[tuple[int, str, str, str], list[float]] = {}
    for record in records:
        for metric, value in record["relative_changes"].items():
            key = (record["tensor"], record["method"], metric)
            grouped.setdefault(key, []).append(float(value))
            layer_key = (int(record["layer"]), *key)
            layer_grouped.setdefault(layer_key, []).append(float(value))
    rows = []
    for (tensor_name, method, metric), values in sorted(grouped.items()):
        rows.append(
            {
                "tensor": tensor_name,
                "method": method,
                "metric": metric,
                "layer_seed_units": len(values),
                "mean_relative_change": statistics.fmean(values),
                "median_relative_change": statistics.median(values),
                "improved_units": sum(value < 0 for value in values),
                "unit_win_rate": statistics.fmean(value < 0 for value in values),
                "minimum_relative_change": min(values),
                "maximum_relative_change": max(values),
                "inferential_status": "training_split_diagnostic_not_a_functional_claim",
            }
        )
    layer_rows = [
        {
            "layer": layer,
            "tensor": tensor_name,
            "method": method,
            "metric": metric,
            "initialization_seeds": len(values),
            "mean_relative_change": statistics.fmean(values),
            "seed_win_rate": statistics.fmean(value < 0 for value in values),
        }
        for (layer, tensor_name, method, metric), values in sorted(layer_grouped.items())
    ]
    p0.write_csv(output_root(cfg) / "training" / "summary.csv", rows)
    p0.write_csv(output_root(cfg) / "training" / "layer_summary.csv", layer_rows)
    p0.atomic_json(
        output_root(cfg) / "training" / "summary.json",
        {
            "schema_version": "wan5b_functional_rvq_training_summary.v1",
            "completed": p0.utc_now(),
            "config_hash": cfg_hash,
            "evaluation_scope": "held-out token subset from the 20 calibration prompts",
            "statistical_warning": "pooled layer-seed diagnostics; exact attention and video benchmarks decide support",
            "rows": rows,
        },
    )
    print(f"[summarize] training_units={len(paths)} records={len(records)}", flush=True)


def main() -> None:
    args = parse_args()
    cfg, cfg_hash, cfg_path = load_config(args.config)
    if args.command in {"prepare", "all"}:
        prepare(cfg, cfg_hash, cfg_path)
    if args.command == "worker":
        worker(cfg, cfg_hash, args.shard_id, args.num_shards)
    if args.command in {"launch", "all"}:
        launch(cfg, cfg_path, args.gpus)
    if args.command in {"assemble", "all"}:
        assemble(cfg, cfg_hash)
    if args.command in {"summarize", "all"}:
        summarize_training(cfg, cfg_hash)


if __name__ == "__main__":
    main()
