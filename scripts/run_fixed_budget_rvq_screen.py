#!/usr/bin/env python3
"""Fixed-budget RVQ innovation screen on the Wan2.2-5B BF16 KV dump.

The script deliberately separates the cheap, prompt-independent cache screen
from video generation.  L8C64 and L16C64 INT4-codebook RVQ are the registered
baselines.  Candidate grouped RVQ methods never exceed their baseline index
budget; sparse residual rescue accounts for its route bitmap and FP16 scale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DIAG_ROOT = REPO_ROOT / "scripts" / "rvq" / "diag"
if str(DIAG_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAG_ROOT))

from run_rvq_diag import train_kmeans_quantvideo  # noqa: E402


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = DEFAULT_DATA / "fixed_budget_innovations_20260721"
DIM = 128


@dataclass(frozen=True)
class GroupMethod:
    name: str
    budget: str
    groups: int
    levels_per_group: int
    transform: str

    @property
    def index_bits(self) -> int:
        return self.groups * self.levels_per_group * 6


GROUP_METHODS = {
    "g4l2_direct": GroupMethod("g4l2_direct", "l8", 4, 2, "direct"),
    "g4l2_hadamard": GroupMethod("g4l2_hadamard", "l8", 4, 2, "hadamard"),
    "g4l4_direct": GroupMethod("g4l4_direct", "l16", 4, 4, "direct"),
    "g4l4_hadamard": GroupMethod("g4l4_hadamard", "l16", 4, 4, "hadamard"),
}
BASE_BITS = {"l8": 48, "l16": 96}


def parse_ints(raw: str) -> list[int]:
    result: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            fields = part.split(":")
            start = int(fields[0] or 0)
            stop = int(fields[1])
            step = int(fields[2]) if len(fields) > 2 and fields[2] else 1
            result.extend(range(start, stop, step))
        else:
            result.append(int(part))
    return sorted(set(result))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("command", choices=["train", "eval", "aggregate"])
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--layers", default="0,6,12,20,29")
    p.add_argument("--heads", default="0,12")
    p.add_argument("--methods", default=",".join(GROUP_METHODS))
    p.add_argument("--sample-size", type=int, default=12000)
    p.add_argument("--kmeans-iters", type=int, default=20)
    p.add_argument("--recon-sample", type=int, default=2048)
    p.add_argument("--query-sample", type=int, default=96)
    p.add_argument("--key-sample", type=int, default=1024)
    p.add_argument("--test-prompts", default="0:10")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scope", choices=("full", "allocation_gate", "depth_gate", "common_gate"), default="full")
    p.add_argument("--gate-baseline", choices=("l8", "l16"), default="l8")
    p.add_argument("--gate-depth", type=int, default=12)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def gpu_type() -> str | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(torch.cuda.current_device())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_run_info(args: argparse.Namespace, command: str) -> None:
    split = json.loads((args.data_root / "prompt_split.json").read_text(encoding="utf-8"))
    config = {
        "command": command,
        "layers": parse_ints(args.layers),
        "heads": parse_ints(args.heads),
        "methods": [x for x in args.methods.split(",") if x],
        "sample_size": args.sample_size,
        "kmeans_iters": args.kmeans_iters,
        "recon_sample": args.recon_sample,
        "query_sample": args.query_sample,
        "key_sample": args.key_sample,
        "test_prompts": parse_ints(args.test_prompts),
        "seed": args.seed,
        "scope": args.scope,
        "budget": {
            "l8": "48 index bits per K or V vector; INT4 centroid values with FP16 per-centroid scales",
            "l16": "96 index bits per K or V vector; INT4 centroid values with FP16 per-centroid scales",
            "sparse_rescue": "route bitmap + residual codes + FP16 scale are included",
        },
        "primary_gate": "prompt-macro QKV_MSE relative reduction >= 10% versus matching baseline",
    }
    config_blob = json.dumps(config, sort_keys=True).encode()
    cfg_path = args.output_root / "config.json"
    if not cfg_path.exists():
        atomic_json(cfg_path, config)
    codebook_dirs = [
        args.data_root / "nonorm_k_l8c64_codebooks",
        args.data_root / "nonorm_v_l8c64_codebooks",
        args.data_root / "nonorm_k_l16c64_codebooks",
        args.data_root / "nonorm_v_l16c64_codebooks",
    ]
    hashes: dict[str, str] = {}
    for directory in codebook_dirs:
        marker = directory / "layer_0_head_0.pt"
        hashes[str(directory)] = sha256_file(marker)
    info = {
        "git_commit": git_commit(),
        "config_path": str(cfg_path),
        "config_hash": hashlib.sha256(config_blob).hexdigest(),
        "codebook_path": [str(x) for x in codebook_dirs],
        "codebook_hash": hashes,
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type(),
        "seed": args.seed,
        "prompt_list": {
            "train": [x["prompt"] for x in split["train"]],
            "test": [split["test"][i]["prompt"] for i in parse_ints(args.test_prompts)],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    atomic_json(args.output_root / "run_info" / f"{command}_shard{args.shard_index}.json", info)


def hadamard(dim: int, device: torch.device | None = None) -> torch.Tensor:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError(f"Hadamard dimension must be power-of-two, got {dim}")
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < dim:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(dim)


def permutation(seed: int) -> torch.Tensor:
    return torch.randperm(DIM, generator=torch.Generator().manual_seed(seed))


def to_groups(x: torch.Tensor, method: GroupMethod, perm: torch.Tensor) -> torch.Tensor:
    y = x.float()
    if method.transform == "hadamard":
        y = y @ hadamard(DIM, device=y.device)
    return y[:, perm.to(y.device)].reshape(y.shape[0], method.groups, DIM // method.groups).contiguous()


def from_groups(y: torch.Tensor, method: GroupMethod, perm: torch.Tensor) -> torch.Tensor:
    flat = y.reshape(y.shape[0], DIM)
    out = torch.empty_like(flat)
    out[:, perm.to(flat.device)] = flat
    if method.transform == "hadamard":
        out = out @ hadamard(DIM, device=out.device)
    return out.contiguous()


def quantize_codebook_int4(codebooks: torch.Tensor) -> torch.Tensor:
    # Runtime RVQ stores one FP16 scale per centroid vector.
    # Clamp after division to the smallest normal FP16 scale.  Otherwise an
    # all-zero/near-zero local centroid underflows its stored scale to zero and
    # produces 0/0 during quantization.
    scale = (codebooks.float().abs().amax(-1, keepdim=True) / 7.0).clamp_min(torch.finfo(torch.float16).tiny).half().float()
    return (torch.round(codebooks.float() / scale).clamp(-7, 7) * scale).contiguous()


def nearest(x: torch.Tensor, c: torch.Tensor, device: torch.device, batch: int = 4096) -> torch.Tensor:
    c_gpu = c.float().to(device)
    c2 = (c_gpu * c_gpu).sum(-1)
    result = []
    for start in range(0, x.shape[0], batch):
        xb = x[start : start + batch].float().to(device)
        dist = (xb * xb).sum(-1, keepdim=True) + c2[None] - 2 * xb @ c_gpu.T
        result.append(torch.argmin(dist, -1))
    return torch.cat(result)


def rvq_reconstruct_depths(x: torch.Tensor, codebooks: torch.Tensor, device: torch.device, depths: set[int]) -> dict[int, torch.Tensor]:
    cb = quantize_codebook_int4(codebooks)
    # Encoding must never mutate the BF16/FP32 reference.  ``float()`` and
    # ``contiguous()`` may both be no-ops for an already-contiguous FP32 input.
    residual = x.to(device=device, dtype=torch.float32).clone(memory_format=torch.contiguous_format)
    cb = cb.to(device)
    recon = torch.zeros_like(residual)
    levels = min(max(depths), cb.shape[0])
    outputs: dict[int, torch.Tensor] = {}
    for level in range(levels):
        ids = nearest(residual, cb[level], device)
        chosen = cb[level, ids]
        recon += chosen
        residual -= chosen
        depth = level + 1
        if depth in depths:
            outputs[depth] = recon.clone()
    return outputs


def rvq_reconstruct(x: torch.Tensor, codebooks: torch.Tensor, device: torch.device, depth: int | None = None) -> torch.Tensor:
    requested = codebooks.shape[0] if depth is None else min(depth, codebooks.shape[0])
    return rvq_reconstruct_depths(x, codebooks, device, {requested})[requested]


def grouped_reconstruct(x: torch.Tensor, payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    method = GROUP_METHODS[payload["method"]]
    perm = payload["permutation"].long()
    grouped = to_groups(x, method, perm)
    codebooks = quantize_codebook_int4(payload["codebooks"].float())
    rec_groups = []
    for group in range(method.groups):
        rec_groups.append(rvq_reconstruct(grouped[:, group], codebooks[:, group], device))
    return from_groups(torch.stack(rec_groups, 1), method, perm)


def layer_path(root: Path, split: str, prompt: int, layer: int) -> Path:
    return root / "kv_dump" / split / str(prompt) / f"layer_{layer}.pt"


def codebook_path(root: Path, method: str, tensor: str, layer: int, head: int) -> Path:
    return root / "codebooks" / method / tensor.lower() / f"layer_{layer}_head_{head}.pt"


def baseline_path(data: Path, budget: str, tensor: str, layer: int, head: int) -> Path:
    return data / f"nonorm_{tensor.lower()}_{budget}c64_codebooks" / f"layer_{layer}_head_{head}.pt"


def sample_train(args: argparse.Namespace, layer: int, head: int, tensor: str) -> torch.Tensor:
    split = json.loads((args.data_root / "prompt_split.json").read_text(encoding="utf-8"))
    take_each = math.ceil(args.sample_size / len(split["train"]))
    chunks = []
    for item in split["train"]:
        pid = int(item["prompt_id"])
        payload = torch.load(layer_path(args.data_root, "train", pid, layer), map_location="cpu", weights_only=False, mmap=True)
        x = payload[tensor][:, head].reshape(-1, DIM).float()
        gen = torch.Generator().manual_seed(args.seed + pid * 1000003 + layer * 1009 + head * 97 + (tensor == "V"))
        chunks.append(x[torch.randperm(x.shape[0], generator=gen)[:take_each]])
    x = torch.cat(chunks)
    return x[: args.sample_size].contiguous()


def train_grouped(x: torch.Tensor, method: GroupMethod, args: argparse.Namespace, tensor: str, layer: int, head: int) -> dict[str, Any]:
    perm = permutation(args.seed)
    groups = to_groups(x, method, perm)
    all_codebooks = []
    residual_history = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for group in range(method.groups):
        residual = groups[:, group].contiguous()
        group_cb, group_hist = [], []
        for level in range(method.levels_per_group):
            centroids, ids, _ = train_kmeans_quantvideo(residual, 64, args.kmeans_iters, device)
            residual = (residual - centroids[ids]).contiguous()
            group_cb.append(centroids)
            group_hist.append(float((residual * residual).mean()))
        all_codebooks.append(torch.stack(group_cb))
        residual_history.append(group_hist)
    return {
        "method": method.name,
        "budget": method.budget,
        "groups": method.groups,
        "levels_per_group": method.levels_per_group,
        "transform": method.transform,
        "index_bits_per_vector": method.index_bits,
        "permutation": perm,
        "codebooks": torch.stack(all_codebooks, 1),
        "tensor": tensor,
        "layer": layer,
        "head": head,
        "train_prompts": 20,
        "sample_size": x.shape[0],
        "seed": args.seed,
        "residual_mse": residual_history,
        "storage": "INT4 centroid values + FP16 scale; uint6 indices",
    }


def selected_units(args: argparse.Namespace) -> list[tuple[int, int]]:
    units = [(layer, head) for layer in parse_ints(args.layers) for head in parse_ints(args.heads)]
    return units[args.shard_index :: args.num_shards]


def command_train(args: argparse.Namespace) -> None:
    methods = [GROUP_METHODS[x] for x in args.methods.split(",") if x]
    for layer, head in selected_units(args):
        for tensor in ("K", "V"):
            pending = [m for m in methods if not codebook_path(args.output_root, m.name, tensor, layer, head).exists()]
            if not pending:
                continue
            x = sample_train(args, layer, head, tensor)
            # Training-query second moment is a deployable K routing statistic.
            if tensor == "K":
                q = sample_train(args, layer, head, "Q")
                cal = args.output_root / "calibration" / f"layer_{layer}_head_{head}.pt"
                cal.parent.mkdir(parents=True, exist_ok=True)
                if not cal.exists():
                    torch.save({"q_second_moment": (q * q).mean(0), "train_prompts": 20, "seed": args.seed}, cal)
            for method in pending:
                out = codebook_path(args.output_root, method.name, tensor, layer, head)
                out.parent.mkdir(parents=True, exist_ok=True)
                payload = train_grouped(x, method, args, tensor, layer, head)
                torch.save(payload, out)
                print(f"[trained] {out}", flush=True)


def affine_quant(x: torch.Tensor, bits: int, group_size: int, affine: bool) -> torch.Tensor:
    shape = x.shape
    if DIM % group_size:
        raise ValueError("group_size must divide 128")
    y = x.float().reshape(-1, DIM // group_size, group_size)
    levels = (1 << bits) - 1
    if affine:
        lo = y.amin(-1, keepdim=True)
        hi = y.amax(-1, keepdim=True)
        scale = ((hi - lo) / levels).clamp_min(1e-8)
        rec = torch.round((y - lo) / scale).clamp(0, levels) * scale + lo
    else:
        if bits == 2:
            scale = y.abs().amax(-1, keepdim=True).clamp_min(1e-8)
            q = torch.round(((y / scale) + 1) * 1.5).clamp(0, 3)
            rec = (q / 1.5 - 1) * scale
        else:
            qmax = (1 << (bits - 1)) - 1
            scale = y.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
            rec = torch.round(y / scale).clamp(-qmax, qmax) * scale
    return rec.reshape(shape).contiguous()


def rescue_fraction(budget: str, bits: int, dim: int = DIM) -> tuple[int, float, int]:
    total = BASE_BITS[budget]
    base_depth = 4 if budget == "l8" else 8
    # One route bit/vector; one FP16 scale per selected vector.
    residual_bits = dim * bits + 16
    fraction = (total - base_depth * 6 - 1) / residual_bits
    return base_depth, max(0.0, fraction), residual_bits


def sparse_rescue(x: torch.Tensor, cb: torch.Tensor, budget: str, bits: int, device: torch.device, q_second: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, float]]:
    depth, fraction, residual_bits = rescue_fraction(budget, bits)
    base = rvq_reconstruct(x, cb, device, depth=depth)
    residual = x.to(device=device, dtype=torch.float32) - base
    score = residual.square()
    if q_second is not None:
        score = score * q_second.to(device=device, dtype=torch.float32)[None]
    score = score.sum(-1)
    count = int(math.floor(fraction * x.shape[0]))
    rec = base.clone()
    if count:
        ids = torch.topk(score, count, sorted=False).indices
        rec[ids] += affine_quant(residual[ids], bits, DIM, affine=False)
    actual = depth * 6 + 1 + (count / x.shape[0]) * residual_bits
    return rec, {"rescue_fraction": count / x.shape[0], "actual_index_and_metadata_bits": actual}


def load_baseline(data: Path, budget: str, tensor: str, layer: int, head: int) -> torch.Tensor:
    return torch.load(baseline_path(data, budget, tensor, layer, head), map_location="cpu", weights_only=False)["codebooks"].float()


def load_deep_codebook(data: Path, tensor: str, layer: int, head: int) -> torch.Tensor:
    if tensor == "V":
        path = data / "value_token_nonorm_v_l32c64_codebooks" / f"layer_{layer}_head_{head}.pt"
    else:
        path = data / "nonorm_k_l16c64_codebooks" / f"layer_{layer}_head_{head}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mode = str(payload.get("input_norm_mode", payload.get("config", {}).get("input_norm_mode", ""))).lower()
    if mode not in {"none", "raw"}:
        raise ValueError(f"Expected raw/no-norm codebook, got {mode}: {path}")
    return payload["codebooks"].float()


def metric_context(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = q.float() @ k.float().T / math.sqrt(DIM)
    p = F.softmax(logits, -1)
    return logits, p, p @ v.float()


def metric_row(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor, meta: dict[str, Any], context: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> dict[str, Any]:
    logits, p, out = metric_context(q, k, v) if context is None else context
    logits_h = q.float() @ kh.float().T / math.sqrt(DIM)
    ph = F.softmax(logits_h, -1)
    out_h = ph @ vh.float()
    return {
        **meta,
        "K_MSE": float(F.mse_loss(kh.float(), k.float())),
        "V_MSE": float(F.mse_loss(vh.float(), v.float())),
        "QK_MSE": float(F.mse_loss(logits_h, logits)),
        "attention_KL": float((p * (p.clamp_min(1e-8).log() - ph.clamp_min(1e-8).log())).sum(-1).mean()),
        "QKV_MSE": float(F.mse_loss(out_h, out)),
        "QKV_cos": float(F.cosine_similarity(out_h, out, -1).mean()),
    }


def eval_one(args: argparse.Namespace, prompt: int, layer: int, head: int, device: torch.device) -> list[dict[str, Any]]:
    saved = torch.load(layer_path(args.data_root, "test", prompt, layer), map_location="cpu", weights_only=False, mmap=True)
    flat = {name: saved[name][:, head].reshape(-1, DIM).float() for name in ("Q", "K", "V")}
    gen = torch.Generator().manual_seed(args.seed + prompt * 1000003 + layer * 1009 + head)
    key_ids = torch.randperm(flat["K"].shape[0], generator=gen)[: args.key_sample]
    q_ids = torch.randperm(flat["Q"].shape[0], generator=gen)[: args.query_sample]
    q, k, v = flat["Q"][q_ids], flat["K"][key_ids], flat["V"][key_ids]
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    context = metric_context(q, k, v)
    if args.scope == "common_gate":
        bases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        rows = []
        for budget in ("l8", "l16"):
            kh = rvq_reconstruct(k, load_baseline(args.data_root, budget, "K", layer, head), device)
            vh = rvq_reconstruct(v, load_baseline(args.data_root, budget, "V", layer, head), device)
            bases[budget] = (kh, vh)
            rows.append(metric_row(q, k, v, kh, vh, {
                "method": f"baseline_{budget}c64_int4", "budget": budget, "family": "baseline",
            }, context))
        for bits in (2, 4):
            for group_size, affine, label in (
                (128, False, "vector_sym"),
                (32, False, "group32_sym"),
                (32, True, "group32_affine"),
            ):
                rows.append(metric_row(
                    q, k, v,
                    affine_quant(k, bits, group_size, affine),
                    affine_quant(v, bits, group_size, affine),
                    {"method": f"direct_{label}_int{bits}", "budget": "control", "family": "common_direct"},
                    context,
                ))
                for budget, (kh, vh) in bases.items():
                    rows.append(metric_row(
                        q, k, v,
                        affine_quant(kh, bits, group_size, affine),
                        affine_quant(vh, bits, group_size, affine),
                        {"method": f"{budget}_then_{label}_int{bits}", "budget": budget, "family": "common_cascade"},
                        context,
                    ))
        for row in rows:
            row.update({"prompt_id": prompt, "layer": layer, "head": head, "key_sample": len(key_ids), "query_sample": len(q_ids)})
        return rows
    if args.scope == "depth_gate":
        budget = args.gate_baseline
        kb = rvq_reconstruct(k, load_baseline(args.data_root, budget, "K", layer, head), device)
        vb = rvq_reconstruct(v, load_baseline(args.data_root, budget, "V", layer, head), device)
        if args.gate_depth > 16:
            raise ValueError("depth_gate currently supports no-norm L16 family prefixes up to depth 16")
        kd = rvq_reconstruct(k, load_baseline(args.data_root, "l16", "K", layer, head), device, args.gate_depth)
        vd = rvq_reconstruct(v, load_baseline(args.data_root, "l16", "V", layer, head), device, args.gate_depth)
        rows = [
            metric_row(q, k, v, kb, vb, {"method": f"baseline_{budget}c64_int4", "budget": budget, "family": "baseline"}, context),
            metric_row(q, k, v, kd, vd, {"method": f"entropy_depth{args.gate_depth}_vs_{budget}", "budget": budget, "family": "entropy_depth_candidate", "depth": args.gate_depth}, context),
        ]
        for row in rows:
            row.update({"prompt_id": prompt, "layer": layer, "head": head, "key_sample": len(key_ids), "query_sample": len(q_ids)})
        return rows
    if args.scope == "allocation_gate":
        k8 = load_baseline(args.data_root, "l8", "K", layer, head)
        v8 = load_baseline(args.data_root, "l8", "V", layer, head)
        kb = rvq_reconstruct(k, k8, device)
        vb = rvq_reconstruct(v, v8, device)
        k16 = load_baseline(args.data_root, "l16", "K", layer, head)
        v16 = load_baseline(args.data_root, "l16", "V", layer, head)
        kc = rvq_reconstruct(k, k16, device, depth=4)
        vc = rvq_reconstruct(v, v16, device, depth=12)
        rows = [
            metric_row(q, k, v, kb, vb, {"method": "baseline_l8c64_int4", "budget": "l8", "family": "baseline"}, context),
            metric_row(q, k, v, kc, vc, {"method": "asym_l8_k4_v12", "budget": "l8", "family": "kv_stage_allocation", "K_depth": 4, "V_depth": 12, "pair_index_bits": 96}, context),
        ]
        for row in rows:
            row.update({"prompt_id": prompt, "layer": layer, "head": head, "key_sample": len(key_ids), "query_sample": len(q_ids)})
        return rows
    bases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    rows = []
    for budget in ("l8", "l16"):
        kcb = load_baseline(args.data_root, budget, "K", layer, head)
        vcb = load_baseline(args.data_root, budget, "V", layer, head)
        kh = rvq_reconstruct(k, kcb, device)
        vh = rvq_reconstruct(v, vcb, device)
        bases[budget] = (kh, vh)
        rows.append(metric_row(q, k, v, kh, vh, {"method": f"baseline_{budget}c64_int4", "budget": budget, "family": "baseline"}, context))
        q_second_path = args.output_root / "calibration" / f"layer_{layer}_head_{head}.pt"
        q_second = torch.load(q_second_path, map_location="cpu", weights_only=True)["q_second_moment"] if q_second_path.exists() else None
        for bits in (2, 4):
            kr, km = sparse_rescue(k, kcb, budget, bits, device, q_second)
            vr, vm = sparse_rescue(v, vcb, budget, bits, device, None)
            rows.append(metric_row(q, k, v, kr, vr, {"method": f"rescue_{budget}_int{bits}_qaware", "budget": budget, "family": "sparse_rescue", **{f"K_{a}": b for a, b in km.items()}, **{f"V_{a}": b for a, b in vm.items()}}, context))
    # K/V stage reallocation holds both uint6 index bits and the number of
    # INT4 centroid scalars constant.  Balanced-retrained rows isolate the
    # effect of the separately trained deeper codebook family.
    k_deep = load_deep_codebook(args.data_root, "K", layer, head)
    # L8 reallocations must use the same 50k-sample L16 family for both K and
    # V.  The separately trained 12k-sample L32 V family is only required when
    # an L16 allocation requests V depth > 16.
    v_l16 = load_baseline(args.data_root, "l16", "V", layer, head)
    v_l32 = load_deep_codebook(args.data_root, "V", layer, head)
    allocations = {
        "asym_l8_k4_v12": ("l8", 4, 12),
        "asym_l8_k6_v10": ("l8", 6, 10),
        "balanced_retrained_l8_k8_v8": ("l8", 8, 8),
        "asym_l8_k10_v6": ("l8", 10, 6),
        "asym_l8_k12_v4": ("l8", 12, 4),
        "asym_l16_k8_v24": ("l16", 8, 24),
        "asym_l16_k12_v20": ("l16", 12, 20),
        "balanced_retrained_l16_k16_v16": ("l16", 16, 16),
    }
    needed_k = {kdepth for _, kdepth, _ in allocations.values()}
    needed_v_l16 = {vdepth for budget, _, vdepth in allocations.values() if budget == "l8"}
    needed_v_l32 = {vdepth for budget, _, vdepth in allocations.values() if budget == "l16"}
    k_alloc = rvq_reconstruct_depths(k, k_deep, device, needed_k)
    v_alloc_l16 = rvq_reconstruct_depths(v, v_l16, device, needed_v_l16)
    v_alloc_l32 = rvq_reconstruct_depths(v, v_l32, device, needed_v_l32)
    for name, (budget, kdepth, vdepth) in allocations.items():
        v_recon = v_alloc_l16[vdepth] if budget == "l8" else v_alloc_l32[vdepth]
        rows.append(metric_row(q, k, v, k_alloc[kdepth], v_recon, {
            "method": name,
            "budget": budget,
            "family": "kv_stage_allocation",
            "K_depth": kdepth,
            "V_depth": vdepth,
            "pair_index_bits": (kdepth + vdepth) * 6,
        }, context))
    for name in [x for x in args.methods.split(",") if x]:
        method = GROUP_METHODS[name]
        kp = torch.load(codebook_path(args.output_root, name, "K", layer, head), map_location="cpu", weights_only=False)
        vp = torch.load(codebook_path(args.output_root, name, "V", layer, head), map_location="cpu", weights_only=False)
        kh = grouped_reconstruct(k, kp, device)
        vh = grouped_reconstruct(v, vp, device)
        rows.append(metric_row(q, k, v, kh, vh, {"method": name, "budget": method.budget, "family": "grouped_rvq", "index_bits": method.index_bits}, context))
    # Controlled common-quantizer error: direct vs RVQ->quantizer cascade.
    for bits in (2, 4):
        for group_size, affine, label in ((128, False, "vector_sym"), (32, False, "group32_sym"), (32, True, "group32_affine")):
            kd = affine_quant(k, bits, group_size, affine)
            vd = affine_quant(v, bits, group_size, affine)
            rows.append(metric_row(q, k, v, kd, vd, {"method": f"direct_{label}_int{bits}", "budget": "control", "family": "common_direct"}, context))
            for budget in ("l8", "l16"):
                kh, vh = bases[budget]
                kc = affine_quant(kh, bits, group_size, affine)
                vc = affine_quant(vh, bits, group_size, affine)
                rows.append(metric_row(q, k, v, kc, vc, {"method": f"{budget}_then_{label}_int{bits}", "budget": budget, "family": "common_cascade"}, context))
    for row in rows:
        row.update({"prompt_id": prompt, "layer": layer, "head": head, "key_sample": len(key_ids), "query_sample": len(q_ids)})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command_eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    units = [(p, l, h) for p in parse_ints(args.test_prompts) for l in parse_ints(args.layers) for h in parse_ints(args.heads)]
    units = units[args.shard_index :: args.num_shards]
    rows = []
    for prompt, layer, head in units:
        rows.extend(eval_one(args, prompt, layer, head, device))
        print(f"[evaluated] prompt={prompt} layer={layer} head={head}", flush=True)
    write_csv(args.output_root / "rows" / f"screen_shard{args.shard_index}.csv", rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def command_aggregate(args: argparse.Namespace) -> None:
    rows: list[dict[str, str]] = []
    for path in sorted((args.output_root / "rows").glob("screen_shard*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    metrics = ["K_MSE", "V_MSE", "QK_MSE", "attention_KL", "QKV_MSE", "QKV_cos"]
    # Prompt is the independent unit: first average layers/heads, then prompts.
    prompt_rows = []
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        for prompt in sorted({int(r["prompt_id"]) for r in subset}):
            group = [r for r in subset if int(r["prompt_id"]) == prompt]
            prompt_rows.append({"method": method, "prompt_id": prompt, **{m: mean([float(r[m]) for r in group]) for m in metrics}})
    write_csv(args.output_root / "prompt_macro.csv", prompt_rows)
    summary = []
    baseline = {}
    for method in sorted({r["method"] for r in prompt_rows}):
        group = [r for r in prompt_rows if r["method"] == method]
        out: dict[str, Any] = {"method": method, "prompts": len(group)}
        for metric in metrics:
            out[metric] = mean([float(r[metric]) for r in group])
        summary.append(out)
        if method.startswith("baseline_"):
            baseline["l8" if "l8c64" in method else "l16"] = out
    for out in summary:
        budget = "l8" if out["method"].startswith(("g4l2", "rescue_l8", "asym_l8", "balanced_retrained_l8")) or out["method"].endswith("_vs_l8") else "l16" if out["method"].startswith(("g4l4", "rescue_l16", "asym_l16", "balanced_retrained_l16")) or out["method"].endswith("_vs_l16") else None
        if budget and budget in baseline:
            for metric in ("K_MSE", "V_MSE", "QK_MSE", "attention_KL", "QKV_MSE"):
                out[f"{metric}_improvement_pct"] = 100 * (baseline[budget][metric] - out[metric]) / baseline[budget][metric]
            out["passes_10pct_primary_gate"] = out["QKV_MSE_improvement_pct"] >= 10.0
    write_csv(args.output_root / "summary.csv", summary)
    atomic_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_run_info(args, args.command)
    if args.command == "train":
        command_train(args)
    elif args.command == "eval":
        command_eval(args)
    else:
        command_aggregate(args)


if __name__ == "__main__":
    main()
