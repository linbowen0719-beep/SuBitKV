#!/usr/bin/env python3
"""Train-only global RVQ-depth allocation and held-out fixed-rate evaluation.

Every layer/head/tensor may use a different prefix of the existing raw
L16C64 RVQ family.  The total number of uint6 indices and INT4 centroid
vectors is held exactly equal to the L8C64 baseline over all 30x24 K/V
groups.  Allocation uses only the residual histories stored by the 20-prompt
training run; the ten test prompts are used only by ``eval``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT / "scripts"))
from run_fixed_budget_rvq_screen import (  # noqa: E402
    DEFAULT_DATA,
    atomic_json,
    layer_path,
    load_baseline,
    metric_context,
    metric_row,
    parse_ints,
    rvq_reconstruct,
    sha256_file,
)


DEFAULT_OUTPUT = DEFAULT_DATA / "global_stage_allocation_l8_20260722"
DIM = 128


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("allocate", "eval", "aggregate"))
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--layers", default="0:30")
    p.add_argument("--heads", default="0:24")
    p.add_argument("--test-prompts", default="0:10")
    p.add_argument("--min-depth", type=int, default=1)
    p.add_argument("--max-depth", type=int, default=16)
    p.add_argument("--mean-depth", type=int, default=8)
    p.add_argument("--query-sample", type=int, default=96)
    p.add_argument("--key-sample", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def deep_path(root: Path, tensor: str, layer: int, head: int) -> Path:
    return root / f"nonorm_{tensor.lower()}_l16c64_codebooks" / f"layer_{layer}_head_{head}.pt"


def training_curve(root: Path, tensor: str, layer: int, head: int) -> list[float]:
    payload = torch.load(deep_path(root, tensor, layer, head), map_location="cpu", weights_only=False)
    curve = [float(x) for x in payload["config"]["residual_mse"]]
    if len(curve) < 16:
        raise ValueError(f"Expected 16-level training curve: {tensor} l={layer} h={head}")
    return curve


def greedy_allocate(curves: dict[str, list[float]], total_stages: int, min_depth: int, max_depth: int) -> dict[str, int]:
    """Globally minimize stored train-MSE curves with a prefix-safe DP.

    The historical function name is retained for the test/import surface.  A
    simple marginal-gain heap is not exact when an unusually useful late RVQ
    level is preceded by a weak level, so this is a multiple-choice knapsack.
    """
    if not 1 <= min_depth <= max_depth:
        raise ValueError("invalid depth bounds")
    n = len(curves)
    if not min_depth * n <= total_stages <= max_depth * n:
        raise ValueError("stage budget is outside the requested bounds")
    keys = sorted(curves)
    previous = np.full(total_stages + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    choices = np.zeros((n, total_stages + 1), dtype=np.uint8)
    for group_index, key in enumerate(keys):
        current = np.full_like(previous, np.inf)
        for depth in range(min_depth, max_depth + 1):
            candidate = previous[:-depth] + curves[key][depth - 1]
            destination = current[depth:]
            improve = candidate < destination
            destination[improve] = candidate[improve]
            selected = choices[group_index, depth:]
            selected[improve] = depth
        previous = current
    if not np.isfinite(previous[total_stages]):
        raise RuntimeError("stage budget has no feasible allocation")
    depths: dict[str, int] = {}
    budget = total_stages
    for group_index in range(n - 1, -1, -1):
        depth = int(choices[group_index, budget])
        if not depth:
            raise RuntimeError("allocation backtracking failed")
        depths[keys[group_index]] = depth
        budget -= depth
    return depths


def allocation_path(args: argparse.Namespace) -> Path:
    return args.output_root / "allocation.json"


def command_allocate(args: argparse.Namespace) -> None:
    curves: dict[str, list[float]] = {}
    for tensor in ("K", "V"):
        for layer in parse_ints(args.layers):
            for head in parse_ints(args.heads):
                curves[f"{tensor}:{layer}:{head}"] = training_curve(args.data_root, tensor, layer, head)
    target = args.mean_depth * len(curves)
    depths = greedy_allocate(curves, target, args.min_depth, args.max_depth)
    baseline_train_mse = 0.0
    for tensor in ("K", "V"):
        for layer in parse_ints(args.layers):
            for head in parse_ints(args.heads):
                p = torch.load(
                    args.data_root / f"nonorm_{tensor.lower()}_l8c64_codebooks" / f"layer_{layer}_head_{head}.pt",
                    map_location="cpu", weights_only=False,
                )
                baseline_train_mse += float(p["config"]["residual_mse"][7])
    candidate_train_mse = sum(curves[k][d - 1] for k, d in depths.items())
    payload = {
        "method": "global_train_mse_waterfill_l16_prefix",
        "selection_split": "20 training prompts only (stored codebook residual_mse histories)",
        "groups": len(curves),
        "target_total_stages": target,
        "actual_total_stages": sum(depths.values()),
        "mean_depth": sum(depths.values()) / len(depths),
        "index_bits_per_vector_group_mean": 6 * sum(depths.values()) / len(depths),
        "depth_histogram": dict(sorted(Counter(depths.values()).items())),
        "baseline_l8_train_mse_sum": baseline_train_mse,
        "candidate_train_mse_sum": candidate_train_mse,
        "predicted_train_improvement_pct": 100 * (baseline_train_mse - candidate_train_mse) / baseline_train_mse,
        "depths": depths,
    }
    atomic_json(allocation_path(args), payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "depths"}, indent=2), flush=True)


def selected_units(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    units = [(p, l, h) for p in parse_ints(args.test_prompts) for l in parse_ints(args.layers) for h in parse_ints(args.heads)]
    return units[args.shard_index :: args.num_shards]


def eval_one(args: argparse.Namespace, prompt: int, layer: int, head: int, allocation: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    saved = torch.load(layer_path(args.data_root, "test", prompt, layer), map_location="cpu", weights_only=False, mmap=True)
    flat = {name: saved[name][:, head].reshape(-1, DIM).float() for name in ("Q", "K", "V")}
    gen = torch.Generator().manual_seed(args.seed + prompt * 1000003 + layer * 1009 + head)
    key_ids = torch.randperm(flat["K"].shape[0], generator=gen)[: args.key_sample]
    query_ids = torch.randperm(flat["Q"].shape[0], generator=gen)[: args.query_sample]
    q = flat["Q"][query_ids].to(device)
    k = flat["K"][key_ids].to(device)
    v = flat["V"][key_ids].to(device)
    context = metric_context(q, k, v)
    kb = rvq_reconstruct(k, load_baseline(args.data_root, "l8", "K", layer, head), device)
    vb = rvq_reconstruct(v, load_baseline(args.data_root, "l8", "V", layer, head), device)
    kd = int(allocation["depths"][f"K:{layer}:{head}"])
    vd = int(allocation["depths"][f"V:{layer}:{head}"])
    kc = rvq_reconstruct(k, load_baseline(args.data_root, "l16", "K", layer, head), device, depth=kd)
    vc = rvq_reconstruct(v, load_baseline(args.data_root, "l16", "V", layer, head), device, depth=vd)
    common = {"prompt_id": prompt, "layer": layer, "head": head, "key_sample": len(key_ids), "query_sample": len(query_ids)}
    return [
        metric_row(q, k, v, kb, vb, {**common, "method": "baseline_l8c64_int4"}, context),
        metric_row(q, k, v, kc, vc, {**common, "method": "global_train_mse_waterfill", "K_depth": kd, "V_depth": vd}, context),
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def command_eval(args: argparse.Namespace) -> None:
    allocation = json.loads(allocation_path(args).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for prompt, layer, head in selected_units(args):
        rows.extend(eval_one(args, prompt, layer, head, allocation, device))
        print(f"[eval] prompt={prompt} layer={layer} head={head}", flush=True)
    write_csv(args.output_root / "rows" / f"shard{args.shard_index}.csv", rows)


def command_aggregate(args: argparse.Namespace) -> None:
    rows: list[dict[str, str]] = []
    for path in sorted((args.output_root / "rows").glob("shard*.csv")):
        rows.extend(csv.DictReader(path.open(newline="", encoding="utf-8")))
    metrics = ("K_MSE", "V_MSE", "QK_MSE", "attention_KL", "QKV_MSE", "QKV_cos")
    prompt_rows: list[dict[str, Any]] = []
    for method in sorted({r["method"] for r in rows}):
        for prompt in parse_ints(args.test_prompts):
            group = [r for r in rows if r["method"] == method and int(r["prompt_id"]) == prompt]
            prompt_rows.append({"method": method, "prompt_id": prompt, **{m: sum(float(r[m]) for r in group) / len(group) for m in metrics}})
    write_csv(args.output_root / "prompt_macro.csv", prompt_rows)
    summary: dict[str, Any] = {"prompts": len(parse_ints(args.test_prompts)), "groups_per_prompt": len(rows) // (2 * len(parse_ints(args.test_prompts)))}
    macro: dict[str, dict[str, float]] = {}
    for method in sorted({r["method"] for r in prompt_rows}):
        group = [r for r in prompt_rows if r["method"] == method]
        macro[method] = {m: sum(float(r[m]) for r in group) / len(group) for m in metrics}
    summary["macro"] = macro
    base, cand = macro["baseline_l8c64_int4"], macro["global_train_mse_waterfill"]
    summary["improvement_pct"] = {m: 100 * (base[m] - cand[m]) / base[m] for m in metrics[:-1]}
    per_prompt = []
    for prompt in parse_ints(args.test_prompts):
        b = next(r for r in prompt_rows if r["method"] == "baseline_l8c64_int4" and r["prompt_id"] == prompt)
        c = next(r for r in prompt_rows if r["method"] == "global_train_mse_waterfill" and r["prompt_id"] == prompt)
        per_prompt.append(100 * (b["QKV_MSE"] - c["QKV_MSE"]) / b["QKV_MSE"])
    gen = torch.Generator().manual_seed(args.seed)
    boots = []
    values = torch.tensor(per_prompt, dtype=torch.float64)
    for _ in range(10000):
        boots.append(float(values[torch.randint(len(values), (len(values),), generator=gen)].mean()))
    summary["QKV_prompt_improvement_pct"] = per_prompt
    summary["QKV_bootstrap_95pct_CI"] = [float(torch.tensor(boots).quantile(.025)), float(torch.tensor(boots).quantile(.975))]
    allocation = json.loads(allocation_path(args).read_text())
    summary["rate_accounting"] = {
        "candidate_total_stages": allocation["actual_total_stages"],
        "baseline_total_stages": allocation["groups"] * 8,
        "candidate_equals_l8_index_and_int4_codebook_budget": allocation["actual_total_stages"] == allocation["groups"] * 8,
    }
    atomic_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def write_run_info(args: argparse.Namespace) -> None:
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg_path = args.output_root / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists(): atomic_json(cfg_path, cfg)
    split = json.loads((args.data_root / "prompt_split.json").read_text(encoding="utf-8"))
    dirs = [args.data_root / f"nonorm_{t}_l{d}c64_codebooks" for t in ("k", "v") for d in (8, 16)]
    info = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "config_path": str(cfg_path),
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "codebook_path": [str(x) for x in dirs],
        "codebook_hash": {str(x): sha256_file(x / "layer_0_head_0.pt") for x in dirs},
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "prompt_list": {s: [x["prompt"] for x in split[s]] for s in ("train", "test")},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": args.command, "shard_index": args.shard_index, "num_shards": args.num_shards,
    }
    atomic_json(args.output_root / "run_info" / f"{args.command}_shard{args.shard_index}.json", info)


def main() -> None:
    args = parse_args(); write_run_info(args)
    {"allocate": command_allocate, "eval": command_eval, "aggregate": command_aggregate}[args.command](args)


if __name__ == "__main__":
    main()
