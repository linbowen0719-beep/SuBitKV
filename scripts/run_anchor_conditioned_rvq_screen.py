#!/usr/bin/env python3
"""Rate-matched anchor-conditioned residual RVQ screen.

Stage-1 keeps C64.  Its code is mapped to one of eight anchor groups; later
stages search C8 inside that group.  L8 uses 13 conditional stages (45 index
bits) and L16 uses 27 (87 bits), financing the larger INT4 codebook from index
bit savings at the registered 80*880-token cache length.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
DIAG_ROOT = SCRIPTS_ROOT / "rvq" / "diag"
for _path in (REPO_ROOT, SCRIPTS_ROOT, DIAG_ROOT):
    if str(_path) not in os.sys.path:
        os.sys.path.insert(0, str(_path))

from run_fixed_budget_rvq_screen import (
    DEFAULT_DATA,
    layer_path,
    load_baseline,
    metric_context,
    metric_row,
    nearest,
    parse_ints,
    quantize_codebook_int4,
    rvq_reconstruct,
    sample_train,
)
from run_rvq_diag import train_kmeans_quantvideo


DEFAULT_OUTPUT = DEFAULT_DATA / "anchor_conditioned_fixed_budget_20260721"
DIM = 128
ANCHORS = 8
LOCAL_C = 8
COND_STAGES = {"l8": 13, "l16": 27}


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("train", "train_tail", "eval", "aggregate"))
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--codebook-root", type=Path, default=None, help="Read trained candidate codebooks from this experiment root")
    p.add_argument("--budgets", default="l8,l16")
    p.add_argument("--tensors", default="K,V", help="Candidate tensors to train; tail training always targets V")
    p.add_argument("--scope", choices=("all", "swap_only"), default="all")
    p.add_argument("--layers", default="0,6,12,20,29")
    p.add_argument("--heads", default="0,12")
    p.add_argument("--test-prompts", default="0:10")
    p.add_argument("--sample-size", type=int, default=12000)
    p.add_argument("--kmeans-iters", type=int, default=20)
    p.add_argument("--query-sample", type=int, default=96)
    p.add_argument("--key-sample", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def artifact_path(args: argparse.Namespace, budget: str, tensor: str, layer: int, head: int) -> Path:
    root = args.codebook_root if args.codebook_root is not None else args.output_root
    return root / "codebooks" / budget / tensor.lower() / f"layer_{layer}_head_{head}.pt"


def tail_path(args: argparse.Namespace, budget: str, layer: int, head: int) -> Path:
    root = args.codebook_root if args.codebook_root is not None else args.output_root
    return root / "v_tail_codebooks" / budget / f"layer_{layer}_head_{head}.pt"


def train_anchor_local_kmeans(
    residual: torch.Tensor,
    token_group: torch.Tensor,
    *,
    max_iters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train all eight C8 local books together without per-shape Triton JIT."""
    device = residual.device
    centroids = []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for group in range(ANCHORS):
        ids = torch.nonzero(token_group == group, as_tuple=False).flatten().cpu()
        if ids.numel() == 0:
            # A stage-1 centroid group can be unused in the finite calibration
            # sample yet appear on a held-out prompt.  Keep a deterministic
            # global-residual fallback book for that rare route.
            ids = torch.arange(residual.shape[0], dtype=torch.long)
        order = ids[torch.randperm(ids.numel(), generator=gen)]
        pick = order.repeat((LOCAL_C + ids.numel() - 1) // ids.numel())[:LOCAL_C].to(device)
        centroids.append(residual[pick].clone())
    c = torch.stack(centroids)
    assigned = torch.zeros(residual.shape[0], dtype=torch.long, device=device)
    flat_group = token_group.long() * LOCAL_C
    for _ in range(max_iters):
        local = c[token_group.long()]
        dist = ((residual[:, None, :] - local) ** 2).sum(-1)
        assigned = torch.argmin(dist, -1)
        flat_ids = flat_group + assigned
        sums = torch.zeros(ANCHORS * LOCAL_C, DIM, device=device)
        sums.index_add_(0, flat_ids, residual)
        counts = torch.bincount(flat_ids, minlength=ANCHORS * LOCAL_C).to(residual.dtype)
        updated = sums.reshape(ANCHORS, LOCAL_C, DIM) / counts.reshape(ANCHORS, LOCAL_C, 1).clamp_min(1)
        nonempty = counts.reshape(ANCHORS, LOCAL_C) > 0
        c = torch.where(nonempty[:, :, None], updated, c)
    return c.detach().cpu(), assigned


def train_one(args: argparse.Namespace, budget: str, tensor: str, layer: int, head: int, device: torch.device) -> dict[str, Any]:
    x = sample_train(args, layer, head, tensor).float()
    stage1 = load_baseline(args.data_root, budget, tensor, layer, head)[:1]
    c1 = quantize_codebook_int4(stage1[0]).to(device)
    ids1 = nearest(x, c1, device)
    residual = x.to(device) - c1[ids1]
    # Cluster the 64 stage-1 centroids into eight deterministic anchor groups.
    _anchor_c, mapping, _ = train_kmeans_quantvideo(stage1[0].float(), ANCHORS, args.kmeans_iters, device)
    mapping = mapping.long().to(device)
    token_group = mapping[ids1]
    stages = []
    history = []
    for stage in range(COND_STAGES[budget]):
        centroids, ids = train_anchor_local_kmeans(
            residual,
            token_group,
            max_iters=args.kmeans_iters,
            seed=args.seed + layer * 100003 + head * 1009 + stage * 17 + (tensor == "V"),
        )
        chosen = centroids.to(device)[token_group.long(), ids]
        residual -= chosen
        stage_hist = [float(residual[token_group == group].square().mean()) for group in range(ANCHORS)]
        stages.append(centroids)
        history.append(stage_hist)
    vectors = 80 * 880
    baseline_depth = 8 if budget == "l8" else 16
    candidate_index_bits = 6 + COND_STAGES[budget] * 3
    centroid_bytes = 64 * DIM // 2 + 64 * 2
    baseline_total = vectors * baseline_depth * 6 / 8 + baseline_depth * centroid_bytes
    candidate_total = vectors * candidate_index_bits / 8 + (1 + COND_STAGES[budget]) * centroid_bytes + 64
    return {
        "budget": budget,
        "tensor": tensor,
        "layer": layer,
        "head": head,
        "stage1_codebook": stage1[0].contiguous(),
        "anchor_mapping": mapping.cpu(),
        "conditional_codebooks": torch.stack(stages),
        "conditional_stages": COND_STAGES[budget],
        "index_bits_per_vector": candidate_index_bits,
        "baseline_index_bits_per_vector": baseline_depth * 6,
        "amortized_total_bytes_at_80x880": candidate_total,
        "baseline_total_bytes_at_80x880": baseline_total,
        "within_budget": candidate_total <= baseline_total,
        "train_prompts": 20,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "residual_history": history,
        "storage": "INT4 centroid values + FP16 scale; uint6 stage1 and uint3 conditional indices",
    }


def reconstruct(x: torch.Tensor, payload: dict[str, Any], device: torch.device, conditional_stages: int | None = None) -> torch.Tensor:
    c1 = quantize_codebook_int4(payload["stage1_codebook"].float()).to(device)
    ids1 = nearest(x, c1, device)
    recon = c1[ids1].clone()
    residual = x.to(device=device, dtype=torch.float32) - recon
    mapping = payload["anchor_mapping"].long().to(device)
    token_group = mapping[ids1]
    codebooks = quantize_codebook_int4(payload["conditional_codebooks"].float()).to(device)
    stages = codebooks.shape[0] if conditional_stages is None else min(conditional_stages, codebooks.shape[0])
    for stage in range(stages):
        for group in range(ANCHORS):
            mask = token_group == group
            if not bool(mask.any()):
                continue
            ids = nearest(residual[mask], codebooks[stage, group], device)
            chosen = codebooks[stage, group, ids]
            recon[mask] += chosen
            residual[mask] -= chosen
    return recon


def train_v_tail_one(args: argparse.Namespace, budget: str, layer: int, head: int, device: torch.device) -> dict[str, Any]:
    x = sample_train(args, layer, head, "V").float()
    base_cb = load_baseline(args.data_root, budget, "V", layer, head)
    depth = 8 if budget == "l8" else 16
    base = rvq_reconstruct(x, base_cb, device, depth)
    residual = x.to(device) - base
    stage1 = quantize_codebook_int4(base_cb[0]).to(device)
    ids1 = nearest(x, stage1, device)
    anchor_file = artifact_path(args, budget, "V", layer, head)
    if anchor_file.exists():
        anchor_payload = torch.load(anchor_file, map_location="cpu", weights_only=False)
        mapping = anchor_payload["anchor_mapping"].long().to(device)
    else:
        _anchor_c, mapping_cpu, _ = train_kmeans_quantvideo(base_cb[0].float(), ANCHORS, args.kmeans_iters, device)
        mapping = mapping_cpu.long().to(device)
    token_group = mapping[ids1]
    centroids, _ids = train_anchor_local_kmeans(
        residual,
        token_group,
        max_iters=args.kmeans_iters,
        seed=args.seed + layer * 100003 + head * 1009 + 7919,
    )
    return {
        "budget": budget,
        "tensor": "V",
        "layer": layer,
        "head": head,
        "anchor_mapping": mapping.cpu(),
        "tail_codebooks": centroids,
        "base_depth": depth,
        "tail_index_bits": 3,
        "train_prompts": 20,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "storage": "INT4 C8 centroid values per 8 anchor groups + FP16 scales; uint3 index",
    }


def reconstruct_v_tail(x: torch.Tensor, base: torch.Tensor, base_cb: torch.Tensor, payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    stage1 = quantize_codebook_int4(base_cb[0]).to(device)
    ids1 = nearest(x, stage1, device)
    mapping = payload["anchor_mapping"].long().to(device)
    token_group = mapping[ids1]
    residual = x.to(device=device, dtype=torch.float32) - base
    codebooks = quantize_codebook_int4(payload["tail_codebooks"].float()).to(device)
    recon = base.clone()
    for group in range(ANCHORS):
        mask = token_group == group
        if not bool(mask.any()):
            continue
        ids = nearest(residual[mask], codebooks[group], device)
        recon[mask] += codebooks[group, ids]
    return recon


def selected_units(args: argparse.Namespace, split: str) -> list[tuple[int, ...]]:
    if split == "train":
        units = [(l, h) for l in parse_ints(args.layers) for h in parse_ints(args.heads)]
    else:
        units = [(p, l, h) for p in parse_ints(args.test_prompts) for l in parse_ints(args.layers) for h in parse_ints(args.heads)]
    return units[args.shard_index :: args.num_shards]


def command_train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for layer, head in selected_units(args, "train"):
        for budget in args.budgets.split(","):
            for tensor in [x.strip().upper() for x in args.tensors.split(",") if x.strip()]:
                if tensor not in {"K", "V"}:
                    raise ValueError(f"Unsupported tensor {tensor}")
                out = artifact_path(args, budget, tensor, layer, head)
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                payload = train_one(args, budget, tensor, layer, head, device)
                if not payload["within_budget"]:
                    raise RuntimeError(f"Candidate exceeds {budget} total budget")
                torch.save(payload, out)
                print(f"[trained] {out}", flush=True)


def command_train_tail(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for layer, head in selected_units(args, "train"):
        for budget in args.budgets.split(","):
            out = tail_path(args, budget, layer, head)
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(train_v_tail_one(args, budget, layer, head, device), out)
            print(f"[trained-tail] {out}", flush=True)


def command_eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for prompt, layer, head in selected_units(args, "eval"):
        saved = torch.load(layer_path(args.data_root, "test", prompt, layer), map_location="cpu", weights_only=False, mmap=True)
        flat = {x: saved[x][:, head].reshape(-1, DIM).float() for x in ("Q", "K", "V")}
        gen = torch.Generator().manual_seed(args.seed + prompt * 1000003 + layer * 1009 + head)
        ids = torch.randperm(flat["K"].shape[0], generator=gen)[: args.key_sample]
        qids = torch.randperm(flat["Q"].shape[0], generator=gen)[: args.query_sample]
        q, k, v = flat["Q"][qids].to(device), flat["K"][ids].to(device), flat["V"][ids].to(device)
        context = metric_context(q, k, v)
        for budget in args.budgets.split(","):
            base_depth = 8 if budget == "l8" else 16
            kb = rvq_reconstruct(k, load_baseline(args.data_root, budget, "K", layer, head), device, base_depth)
            v_base_cb = load_baseline(args.data_root, budget, "V", layer, head)
            vb = rvq_reconstruct(v, v_base_cb, device, base_depth)
            kp = torch.load(artifact_path(args, budget, "K", layer, head), map_location="cpu", weights_only=False)
            common = {"prompt_id": prompt, "layer": layer, "head": head, "budget": budget}
            if args.scope == "swap_only":
                tail = torch.load(tail_path(args, budget, layer, head), map_location="cpu", weights_only=False)
                kc_swap = reconstruct(k, kp, device, conditional_stages=COND_STAGES[budget] - 1)
                vc_tail = reconstruct_v_tail(v, vb, v_base_cb, tail, device)
                k_bits = 6 + (COND_STAGES[budget] - 1) * 3
                v_bits = base_depth * 6 + 3
                rows.append(metric_row(q, k, v, kb, vb, {**common, "method": f"baseline_{budget}c64_int4", "family": "baseline"}, context))
                rows.append(metric_row(q, k, v, kc_swap, vc_tail, {**common, "method": f"anchor_k_vtail_{budget}", "family": "anchor_conditioned_k_vtail", "K_index_bits": k_bits, "V_index_bits": v_bits}, context))
                continue
            vp = torch.load(artifact_path(args, budget, "V", layer, head), map_location="cpu", weights_only=False)
            kc, vc = reconstruct(k, kp, device), reconstruct(v, vp, device)
            rows.append(metric_row(q, k, v, kb, vb, {**common, "method": f"baseline_{budget}c64_int4", "family": "baseline"}, context))
            rows.append(metric_row(q, k, v, kc, vc, {**common, "method": f"anchor_cond_{budget}", "family": "anchor_conditioned", "index_bits": kp["index_bits_per_vector"]}, context))
            rows.append(metric_row(q, k, v, kc, vb, {**common, "method": f"anchor_k_only_{budget}", "family": "anchor_conditioned_k_only", "K_index_bits": kp["index_bits_per_vector"]}, context))
            rows.append(metric_row(q, k, v, kb, vc, {**common, "method": f"anchor_v_only_{budget}", "family": "anchor_conditioned_v_only", "V_index_bits": vp["index_bits_per_vector"]}, context))
            tail_file = tail_path(args, budget, layer, head)
            if tail_file.exists():
                tail = torch.load(tail_file, map_location="cpu", weights_only=False)
                kc_swap = reconstruct(k, kp, device, conditional_stages=COND_STAGES[budget] - 1)
                vc_tail = reconstruct_v_tail(v, vb, load_baseline(args.data_root, budget, "V", layer, head), tail, device)
                k_bits = 6 + (COND_STAGES[budget] - 1) * 3
                v_bits = (8 if budget == "l8" else 16) * 6 + 3
                rows.append(metric_row(q, k, v, kc_swap, vc_tail, {**common, "method": f"anchor_k_vtail_{budget}", "family": "anchor_conditioned_k_vtail", "K_index_bits": k_bits, "V_index_bits": v_bits}, context))
        print(f"[evaluated] p={prompt} l={layer} h={head}", flush=True)
    write_csv(args.output_root / "rows" / f"shard{args.shard_index}.csv", rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def command_aggregate(args: argparse.Namespace) -> None:
    rows = []
    for path in sorted((args.output_root / "rows").glob("shard*.csv")):
        rows.extend(csv.DictReader(path.open()))
    metrics = ("K_MSE", "V_MSE", "QK_MSE", "attention_KL", "QKV_MSE", "QKV_cos")
    prompt_rows = []
    for method in sorted({r["method"] for r in rows}):
        for prompt in sorted({int(r["prompt_id"]) for r in rows if r["method"] == method}):
            group = [r for r in rows if r["method"] == method and int(r["prompt_id"]) == prompt]
            prompt_rows.append({"method": method, "prompt_id": prompt, **{m: sum(float(r[m]) for r in group) / len(group) for m in metrics}})
    write_csv(args.output_root / "prompt_macro.csv", prompt_rows)
    summary = []
    for method in sorted({r["method"] for r in prompt_rows}):
        group = [r for r in prompt_rows if r["method"] == method]
        out = {"method": method, "prompts": len(group), **{m: sum(float(r[m]) for r in group) / len(group) for m in metrics}}
        if method.startswith(("anchor_cond_", "anchor_k_only_", "anchor_v_only_", "anchor_k_vtail_")):
            budget = method.rsplit("_", 1)[-1]
            base = [r for r in prompt_rows if r["method"] == f"baseline_{budget}c64_int4"]
            for m in metrics[:-1]:
                b = sum(float(r[m]) for r in base) / len(base)
                out[f"{m}_improvement_pct"] = 100 * (b - out[m]) / b
            out["passes_10pct_primary_gate"] = out["QKV_MSE_improvement_pct"] >= 10
        summary.append(out)
    write_csv(args.output_root / "summary.csv", summary)
    print(json.dumps(summary, indent=2), flush=True)


def write_run_info(args: argparse.Namespace) -> None:
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg_path = args.output_root / "config.json"; cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists(): cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    split = json.loads((args.data_root / "prompt_split.json").read_text())
    info = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "config_path": str(cfg_path), "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "codebook_path": str(args.output_root / "codebooks"), "codebook_hash": "per-file payload provenance",
        "hostname": socket.gethostname(), "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed, "prompt_list": {x: [z["prompt"] for z in split[x]] for x in ("train", "test")},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    p = args.output_root / "run_info" / f"{args.command}_shard{args.shard_index}.json"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(info, indent=2, ensure_ascii=False))


def main() -> None:
    args = args_parser(); write_run_info(args)
    if args.command == "train": command_train(args)
    elif args.command == "train_tail": command_train_tail(args)
    elif args.command == "eval": command_eval(args)
    else: command_aggregate(args)


if __name__ == "__main__": main()
