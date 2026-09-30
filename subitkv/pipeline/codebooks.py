"""Offline residual K-means codebooks, one independent model layer/head at a time."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def split_path(output_dir: Path) -> Path:
    return output_dir / "prompt_split.json"


def split_items(output_dir: Path, split_name: str) -> list[dict[str, Any]]:
    with split_path(output_dir).open("r", encoding="utf-8") as f:
        split = json.load(f)
    return split[split_name]


def layer_file(output_dir: Path, split_name: str, prompt_id: int, layer: int) -> Path:
    return output_dir / "kv_dump" / split_name / str(prompt_id) / f"layer_{layer}.pt"


def sample_tensor_vectors(
    output_dir: Path,
    layer: int,
    head: int,
    tensor_name: str,
    sample_size: int,
    seed: int,
    sample_seed_mode: str = "standard",
) -> "Any":
    import torch

    items = split_items(output_dir, "train")
    per_prompt = max(1, math.ceil(sample_size / max(1, len(items))))
    samples = []
    for item in items:
        payload = torch.load(
            layer_file(output_dir, "train", int(item["prompt_id"]), layer),
            map_location="cpu",
            weights_only=False,
        )
        tensor = payload[tensor_name][:, head].reshape(
            -1, payload[tensor_name].shape[-1]
        )
        if sample_seed_mode == "grouped_dim_v1":
            tensor_offset = 97 if tensor_name == "V" else 0
            worker_seed = seed + layer * 100000 + head * 1000 + tensor_offset
            prompt_seed = (
                worker_seed + int(item["prompt_id"]) * 1000003 + layer * 1009 + head
            )
        else:
            prompt_seed = seed + layer * 100000 + head * 1000 + int(item["prompt_id"])
        generator = torch.Generator(device="cpu").manual_seed(prompt_seed)
        take = min(per_prompt, tensor.shape[0])
        indices = torch.randperm(tensor.shape[0], generator=generator)[:take]
        samples.append(tensor[indices].float())
    x = torch.cat(samples, dim=0)
    if x.shape[0] > sample_size:
        if sample_seed_mode == "grouped_dim_v1":
            tensor_offset = 97 if tensor_name == "V" else 0
            worker_seed = seed + layer * 100000 + head * 1000 + tensor_offset
            final_seed = worker_seed + layer * 100000 + head
        else:
            final_seed = seed + layer * 100000 + head
        generator = torch.Generator(device="cpu").manual_seed(final_seed)
        indices = torch.randperm(x.shape[0], generator=generator)[:sample_size]
        x = x[indices]
    return x.contiguous()


def train_kmeans_triton(
    x: "Any", k: int, max_iters: int, device: "Any"
) -> tuple["Any", "Any", str]:
    import torch

    from subitkv.kmeans.kmeans_euclid import batch_kmeans_Euclid

    xb = x.to(device=device, dtype=torch.float32).unsqueeze(0).contiguous()
    cluster_ids, centroids, _cluster_sizes, _iters = batch_kmeans_Euclid(
        xb, k, max_iters=max_iters, tol=1e-4, verbose=False
    )
    return (
        centroids[0].detach().cpu().float(),
        cluster_ids[0].detach().cpu().long(),
        "Triton KMeans",
    )


def train_kmeans_faiss(
    x: "Any", k: int, max_iters: int, gpu: int
) -> tuple["Any", "Any", str]:
    import faiss
    import torch

    if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1:
        raise RuntimeError("FAISS GPU is not available")
    data = x.detach().cpu().numpy().astype("float32")
    d = data.shape[1]
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = 0
    index = faiss.GpuIndexFlatL2(res, d, cfg)
    km = faiss.Clustering(d, k)
    km.niter = max_iters
    km.train(data, index)
    centroids = faiss.vector_to_array(km.centroids).reshape(k, d)
    index.reset()
    index.add(centroids.astype("float32"))
    _dist, ids = index.search(data, 1)
    return (
        torch.from_numpy(centroids).float(),
        torch.from_numpy(ids[:, 0]).long(),
        "FAISS GPU KMeans",
    )


def train_kmeans_sklearn(
    x: "Any", k: int, max_iters: int, seed: int
) -> tuple["Any", "Any", str]:
    import torch
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=k,
        max_iter=max_iters,
        batch_size=8192,
        n_init=1,
        random_state=seed,
        reassignment_ratio=0.01,
    )
    ids = km.fit_predict(x.detach().cpu().numpy().astype("float32"))
    return (
        torch.from_numpy(km.cluster_centers_).float(),
        torch.from_numpy(ids).long(),
        "sklearn MiniBatchKMeans",
    )


def train_one(work, calibration, tensor_name, layer, head, gpu):
    import torch

    c = calibration
    path = (
        work
        / f"nonorm_{tensor_name.lower()}_l16c64_codebooks"
        / f"layer_{layer}_head_{head}.pt"
    )
    if path.exists():
        return
    x = sample_tensor_vectors(
        work, layer, head, tensor_name, c["sample_size"], c["seed"]
    )
    norm = c["input_norm_mode"]
    residual = x.float().contiguous()
    if norm != "none":
        residual = residual / residual.square().mean(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)
    codebooks, errors, backends = [], [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for level in range(c["levels"]):
        try:
            centroids, ids, backend = train_kmeans_triton(
                residual, c["centroids"], c["kmeans_iters"], device
            )
        except Exception as exc:
            print(f"Triton K-means unavailable: {exc}", flush=True)
            try:
                centroids, ids, backend = train_kmeans_faiss(
                    residual, c["centroids"], c["kmeans_iters"], gpu
                )
            except Exception as exc:
                print(f"FAISS unavailable: {exc}; using sklearn", flush=True)
                centroids, ids, backend = train_kmeans_sklearn(
                    residual, c["centroids"], c["kmeans_iters"], c["seed"] + level
                )
        residual = (residual.float() - centroids[ids].float()).contiguous()
        errors.append(float(residual.square().mean()))
        backends.append(backend)
        codebooks.append(centroids)
    payload = dict(
        codebooks=torch.stack(codebooks).float(),
        input_norm_mode=norm,
        config=dict(
            layer=layer,
            head=head,
            levels=c["levels"],
            codebook_size=c["centroids"],
            sample_size=len(x),
            sample_seed_mode="standard",
            sample_manifest_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            kmeans_max_iters=c["kmeans_iters"],
            backends=backends,
            backend_priority=[
                "Triton KMeans",
                "FAISS GPU KMeans",
                "sklearn MiniBatchKMeans",
            ],
            residual_mse=errors,
            input_norm_mode=norm,
            normalization="none/raw vector" if norm == "none" else "per-vector RMS",
            train_objective="recon",
            tensor=tensor_name,
            key_type="pre_rope" if tensor_name == "K" else None,
            value_type="pre_prope" if tensor_name == "V" else None,
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    print(f"Saved {path}", flush=True)


def train(work, config, tensor_name):
    """Partition layer/head tasks across the configured visible GPU devices."""
    from .common import REPO, write_json

    work = Path(work)
    m = config["model_config"]
    tasks = [
        (layer, head) for layer in range(m["layers"]) for head in range(m["heads"])
    ]
    gpus = config["runtime"]["gpus"]
    processes = []
    try:
        for index, gpu in enumerate(gpus):
            selected = tasks[index :: len(gpus)]
            if not selected:
                continue
            job = work / f"{tensor_name}_worker_{gpu}.json"
            write_json(
                job,
                dict(
                    work=str(work),
                    calibration=config["calibration"],
                    tensor_name=tensor_name,
                    tasks=selected,
                    gpu=gpu,
                ),
            )
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO))
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-m", "subitkv.pipeline.codebooks", str(job)],
                    cwd=REPO,
                    env=env,
                )
            )
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"Codebook workers failed: {codes}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text())
    for layer, head in job["tasks"]:
        train_one(
            Path(job["work"]),
            job["calibration"],
            job["tensor_name"],
            layer,
            head,
            job["gpu"],
        )
