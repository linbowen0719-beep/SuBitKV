#!/usr/bin/env python3
"""Diagnose why lower unweighted KV MSE can worsen exact attention replay."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402
import analyze_wan5b_p1_compact as compact  # noqa: E402
import analyze_wan5b_p1_exact_replay as exact  # noqa: E402


def load_configs(path: Path) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    cfg, cfg_hash = p0.parse_config(path)
    exact_path = p0.resolve_path(cfg["exact_config"])
    exact_cfg, exact_hash = p0.parse_config(exact_path)
    return cfg, cfg_hash, exact_cfg, exact_hash


def output_root(cfg: dict[str, Any]) -> Path:
    return p0.resolve_path(cfg["output_dir"])


def exact_root(exact_cfg: dict[str, Any]) -> Path:
    return p0.resolve_path(exact_cfg["output_dir"])


def unit_paths(root: Path, index: int) -> dict[str, Path]:
    base = root / "diagnostics" / f"capture_{index:02d}"
    return {
        "metrics": base / "proxy_gap_metrics.csv",
        "done": base / "done.json",
    }


def unpack_runtime(stages: Sequence[dict[str, Any]]) -> list[torch.Tensor]:
    result = []
    for stage in stages:
        signed = exact.unpack_uint4(stage["q_packed"], stage["q_shape"]).to(torch.int16) - 8
        result.append(
            (signed.to(torch.bfloat16) * stage["scale"].to(torch.bfloat16)).contiguous()
        )
    return result


def prompt_metadata(capture_path: Path) -> tuple[int, str]:
    info_path = capture_path.parents[3] / "run_info.json"
    if not info_path.exists():
        return 0, "unknown_capture_prompt"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    prompts = info.get("prompt_list", [])
    return (
        int(info.get("prompt_id", info.get("prompt_start", 0))),
        str(prompts[0]) if prompts else "unknown_capture_prompt",
    )


def proxy_gap_metrics(
    *,
    query: torch.Tensor,
    key_reference: torch.Tensor,
    value_reference: torch.Tensor,
    candidates: dict[str, tuple[torch.Tensor, torch.Tensor]],
    history_tokens: int,
    query_batch_size: int,
) -> dict[str, dict[str, float]]:
    """Decompose K-only and V-only errors before any camera output transform."""
    methods = list(candidates)
    dim = int(query.shape[-1])
    key_count = int(key_reference.shape[-2])
    query_count = int(query.shape[-2])
    scale = dim**-0.5
    accum = {
        method: {
            "reference_output_energy": 0.0,
            "V_output_error_energy": 0.0,
            "V_diagonal_error_energy": 0.0,
            "V_mass_weighted_token_error_energy": 0.0,
            "V_reference_history_mass": 0.0,
            "reference_logit_energy": 0.0,
            "K_logit_error_energy": 0.0,
            "K_probability_error_energy": 0.0,
            "K_output_error_energy": 0.0,
            "K_diagonal_value_error_energy": 0.0,
        }
        for method in methods
    }
    static: dict[str, dict[str, float | torch.Tensor]] = {}
    for method, (key_hat, value_hat) in candidates.items():
        delta_k = (key_hat[..., :history_tokens, :] - key_reference[..., :history_tokens, :]).float()
        delta_v = (value_hat[..., :history_tokens, :] - value_reference[..., :history_tokens, :]).float()
        k_error = float(torch.sum(delta_k**2).item())
        v_error = float(torch.sum(delta_v**2).item())
        k_mean_energy = float(torch.sum(delta_k.mean(dim=-2) ** 2).item())
        v_mean_energy = float(torch.sum(delta_v.mean(dim=-2) ** 2).item())
        static[method] = {
            "delta_v": delta_v,
            "V_token_error": torch.sum(delta_v**2, dim=-1),
            "uniform_K_error_per_dim": k_error / max(history_tokens * dim, 1),
            "uniform_V_error_per_dim": v_error / max(history_tokens * dim, 1),
            "K_mean_error_fraction": k_mean_energy / max(k_error / history_tokens, 1e-20),
            "V_mean_error_fraction": v_mean_energy / max(v_error / history_tokens, 1e-20),
        }
    value_norm = torch.sum(value_reference.float() ** 2, dim=-1)
    for start in range(0, query_count, query_batch_size):
        end = min(query_count, start + query_batch_size)
        q = query[..., start:end, :].float()
        logits_reference = torch.matmul(q, key_reference.float().transpose(-1, -2)) * scale
        probs_reference = torch.softmax(logits_reference, dim=-1)
        reference_output = torch.matmul(probs_reference, value_reference.float())
        reference_energy = float(torch.sum(reference_output**2).item())
        reference_logit_energy = float(torch.sum(logits_reference**2).item())
        probs_history = probs_reference[..., :history_tokens]
        for method, (key_hat, _value_hat) in candidates.items():
            record = accum[method]
            record["reference_output_energy"] += reference_energy
            record["reference_logit_energy"] += reference_logit_energy
            delta_v = static[method]["delta_v"]
            token_error = static[method]["V_token_error"]
            assert isinstance(delta_v, torch.Tensor) and isinstance(token_error, torch.Tensor)
            v_output_error = torch.matmul(probs_history, delta_v)
            record["V_output_error_energy"] += float(torch.sum(v_output_error**2).item())
            record["V_diagonal_error_energy"] += float(
                torch.sum(torch.matmul(probs_history**2, token_error.unsqueeze(-1))).item()
            )
            record["V_mass_weighted_token_error_energy"] += float(
                torch.sum(torch.matmul(probs_history, token_error.unsqueeze(-1))).item()
            )
            record["V_reference_history_mass"] += float(torch.sum(probs_history).item())

            logits_hat = torch.matmul(q, key_hat.float().transpose(-1, -2)) * scale
            logit_delta = logits_hat - logits_reference
            probs_hat = torch.softmax(logits_hat, dim=-1)
            probability_delta = probs_hat - probs_reference
            k_output_error = torch.matmul(probability_delta, value_reference.float())
            record["K_logit_error_energy"] += float(torch.sum(logit_delta**2).item())
            record["K_probability_error_energy"] += float(torch.sum(probability_delta**2).item())
            record["K_output_error_energy"] += float(torch.sum(k_output_error**2).item())
            record["K_diagonal_value_error_energy"] += float(
                torch.sum(torch.matmul(probability_delta**2, value_norm.unsqueeze(-1))).item()
            )
    result = {}
    for method in methods:
        record = accum[method]
        uniform_v = float(static[method]["uniform_V_error_per_dim"])
        mass_weighted_v = record["V_mass_weighted_token_error_energy"] / max(
            record["V_reference_history_mass"] * dim, 1e-20
        )
        result[method] = {
            "uniform_K_error_per_dim": float(static[method]["uniform_K_error_per_dim"]),
            "uniform_V_error_per_dim": uniform_v,
            "K_mean_error_fraction": float(static[method]["K_mean_error_fraction"]),
            "V_mean_error_fraction": float(static[method]["V_mean_error_fraction"]),
            "V_mass_weighted_error_per_dim": mass_weighted_v,
            "V_mass_weighted_vs_uniform": mass_weighted_v / max(uniform_v, 1e-20),
            "V_output_error_relative_energy": record["V_output_error_energy"] / max(
                record["reference_output_energy"], 1e-20
            ),
            "V_cross_token_coherence_multiplier": record["V_output_error_energy"] / max(
                record["V_diagonal_error_energy"], 1e-20
            ),
            "K_logit_error_per_pair": record["K_logit_error_energy"] / max(
                query_count * key_count, 1
            ),
            "K_logit_error_relative_energy": record["K_logit_error_energy"] / max(
                record["reference_logit_energy"], 1e-20
            ),
            "K_probability_error_per_query": record["K_probability_error_energy"] / max(
                query_count, 1
            ),
            "K_output_error_relative_energy": record["K_output_error_energy"] / max(
                record["reference_output_energy"], 1e-20
            ),
            "K_value_coupling_multiplier": record["K_output_error_energy"] / max(
                record["K_diagonal_value_error_energy"], 1e-20
            ),
        }
    return result


def run_capture(
    cfg: dict[str, Any], cfg_hash: str, exact_cfg: dict[str, Any], exact_hash: str,
    index: int, capture_path: Path, device: torch.device,
) -> None:
    paths = unit_paths(output_root(cfg), index)
    if all(path.exists() for path in paths.values()):
        return
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite incomplete diagnostic: {existing}")
    started = time.time()
    raw = torch.load(capture_path, map_location="cpu", weights_only=False)
    metadata = exact.validate_capture(raw)
    tensors, evidence, validation, query_output_matrix, _full_kv, _qp, _kp = exact.positional_replay(raw, device)
    del query_output_matrix
    if not validation["exact_transform_replay"]:
        raise RuntimeError("Exact positional replay validation failed")
    history_tokens = int(metadata["cache_tokens"])
    layer = int(metadata["layer"])
    prompt_id, prompt = prompt_metadata(capture_path)
    codebook_path = exact_root(exact_cfg) / "exact_replay" / f"capture_{index:02d}" / "calibrated_int4_codebooks.pt"
    saved = torch.load(codebook_path, map_location="cpu", weights_only=False)
    if saved.get("config_hash") != exact_hash:
        raise ValueError(f"Exact codebook config mismatch: {codebook_path}")
    rows = []
    methods = [str(value) for value in cfg["methods"]]
    for seed in [int(value) for value in exact_cfg["initialization_seeds"]]:
        method_tensors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for method in methods:
            k_codebooks = unpack_runtime(saved["packed_codebooks"][seed]["K"][method])
            v_codebooks = unpack_runtime(saved["packed_codebooks"][seed]["V"][method])
            k_history = exact.reconstruct_history(
                tensors["key_pre"][..., :history_tokens, :], k_codebooks, device,
                int(exact_cfg["rvq"]["assignment_batch_size"]),
            )
            v_history = exact.reconstruct_history(
                tensors["value_pre"][..., :history_tokens, :], v_codebooks, device,
                int(exact_cfg["rvq"]["assignment_batch_size"]),
            )
            standard_k, standard_v, prope_k, prope_v, _history_kv = exact.candidate_positional_tensors(
                tensors, evidence, history_tokens, k_history, v_history
            )
            method_tensors[method] = (standard_k, standard_v, prope_k, prope_v)
        halves = {
            "standard_rope": (
                tensors["query_post_rope"], tensors["key_post_rope"], tensors["value_post_rope"], 0, 1,
            ),
            "camera_prope": (
                tensors["query_post_prope"], tensors["key_post_prope"], tensors["value_post_prope"], 2, 3,
            ),
        }
        for half_name, (query, key_ref, value_ref, key_slot, value_slot) in halves.items():
            for head in range(p0.HEADS):
                candidates = {
                    method: (
                        method_tensors[method][key_slot][:, head : head + 1],
                        method_tensors[method][value_slot][:, head : head + 1],
                    )
                    for method in methods
                }
                metrics = proxy_gap_metrics(
                    query=query[:, head : head + 1],
                    key_reference=key_ref[:, head : head + 1],
                    value_reference=value_ref[:, head : head + 1],
                    candidates=candidates,
                    history_tokens=history_tokens,
                    query_batch_size=int(cfg["query_batch_size"]),
                )
                for method, values in metrics.items():
                    rows.append(
                        {
                            "study_id": cfg["study_id"], "config_hash": cfg_hash,
                            "hypothesis_id": cfg.get("hypothesis_id", "P1-D1"),
                            "diagnostic_status": "posthoc_explanatory",
                            "prompt_id": prompt_id, "prompt": prompt,
                            "capture_index": index, "layer": layer, "initialization_seed": seed,
                            "attention_half": half_name, "head": head, "method": method,
                            "metric_scope": "pre_camera_output_transform",
                            **values,
                        }
                    )
    p0.write_csv(paths["metrics"], rows)
    p0.atomic_json(
        paths["done"],
        {
            "completed": p0.utc_now(), "config_hash": cfg_hash, "capture_index": index,
            "layer": layer, "rows": len(rows), "seconds": time.time() - started,
            "source_capture_sha256": p0.sha256_file(capture_path),
            "source_codebook_sha256": p0.sha256_file(codebook_path),
        },
    )
    print(f"[diagnostic] capture={index} layer={layer} rows={len(rows)} seconds={time.time()-started:.1f}", flush=True)


def cmd_prepare(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash, exact_cfg, exact_hash = load_configs(cfg_path)
    source_root = exact_root(exact_cfg)
    source_summary_path = source_root / "summary.json"
    source_manifest_path = source_root / "exact_replay" / "codebook_manifest.json"
    if not source_summary_path.exists() or not source_manifest_path.exists():
        raise FileNotFoundError("Selected-schedule exact replay must be complete")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("P1-S3_supported") is not False:
        raise ValueError("P1-D1 is only authorized after the preregistered P1-S3 failure")
    root = output_root(cfg)
    info_path = root / "run_info.json"
    if root.exists() and not info_path.exists():
        raise FileExistsError(f"Refusing existing output without run_info: {root}")
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("config_hash") != cfg_hash:
            raise ValueError("Existing diagnostic config hash mismatch")
        print(f"[prepare] already prepared root={root}")
        return
    exact_info = json.loads((source_root / "run_info.json").read_text(encoding="utf-8"))
    registry = p0.resolve_path(cfg["hypothesis_registry"])
    info = {
        "git_commit": p0.git_commit(), "config_path": str(cfg_path), "config_hash": cfg_hash,
        "codebook_path": str(source_manifest_path.resolve()),
        "codebook_hash": p0.sha256_file(source_manifest_path),
        "hostname": p0.socket.gethostname(), "gpu_type": p0.gpu_name(),
        "seed": int(cfg["seed"]), "prompt_list": exact_info["prompt_list"],
        "timestamp": p0.utc_now(), "source_exact_config_hash": exact_hash,
        "source_exact_summary": {"path": str(source_summary_path.resolve()), "sha256": p0.sha256_file(source_summary_path)},
        "hypothesis_registry": str(registry.resolve()),
        "hypothesis_registry_hash": p0.sha256_file(registry),
        "statistical_unit": "prompt; layers, seeds, heads and attention halves averaged within prompt",
        "diagnostic_status": "posthoc_explanatory_not_confirmatory",
        "source_data_read_only": True,
    }
    p0.atomic_json(info_path, info)
    print(f"[prepare] root={root} config_hash={cfg_hash}", flush=True)


def cmd_worker(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash, exact_cfg, exact_hash = load_configs(cfg_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tasks = exact.capture_tasks(exact_cfg)
    selected = [task for i, task in enumerate(tasks) if i % args.num_shards == args.shard_id]
    print(f"[worker] shard={args.shard_id}/{args.num_shards} tasks={len(selected)} gpu={torch.cuda.get_device_name(0)}", flush=True)
    for task in selected:
        run_capture(cfg, cfg_hash, exact_cfg, exact_hash, *task, torch.device("cuda:0"))


def cmd_launch(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, _cfg_hash, _exact_cfg, _exact_hash = load_configs(cfg_path)
    gpus = [int(value) for value in (args.gpus.split(",") if args.gpus else cfg["gpus"])]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError(f"Expected four distinct GPUs, got {gpus}")
    launch_id = p0.utc_now().replace(":", "").replace("+", "_")
    log_dir = output_root(cfg) / "logs" / f"diagnostic_{launch_id}"
    log_dir.mkdir(parents=True, exist_ok=False)
    processes = []
    for shard, gpu in enumerate(gpus):
        path = log_dir / f"gpu_{gpu}_shard_{shard}.log"
        handle = path.open("x", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            str(cfg["python"]), str(Path(__file__).resolve()), "worker", "--config", str(cfg_path),
            "--shard-id", str(shard), "--num-shards", str(len(gpus)),
        ]
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((gpu, process, handle, path))
        print(f"[launch] gpu={gpu} pid={process.pid} log={path}", flush=True)
    last_report = 0.0
    try:
        while any(process.poll() is None for _gpu, process, _handle, _path in processes):
            if time.time() - last_report >= 10:
                print("[monitor] " + ", ".join(
                    f"gpu{gpu}:pid{process.pid}:{'running' if process.poll() is None else process.returncode}"
                    for gpu, process, _handle, _path in processes
                ), flush=True)
                last_report = time.time()
            time.sleep(2)
    finally:
        for _gpu, _process, handle, _path in processes:
            handle.close()
    failed = [item for item in processes if item[1].returncode != 0]
    if failed:
        for gpu, process, _handle, path in failed:
            print(f"[failed] gpu={gpu} rc={process.returncode} log={path}\n{path.read_text(encoding='utf-8')[-12000:]}", flush=True)
        raise RuntimeError(f"{len(failed)} diagnostic worker(s) failed; logs preserved at {log_dir}")
    print(f"[launch] all workers complete logs={log_dir}", flush=True)


def cmd_merge(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash, exact_cfg, _exact_hash = load_configs(cfg_path)
    root = output_root(cfg)
    expected = len(exact.capture_tasks(exact_cfg))
    done = list((root / "diagnostics").glob("capture_*/done.json"))
    if len(done) != expected:
        raise RuntimeError(f"Incomplete diagnostics: {len(done)}/{expected}")
    rows = p0.merge_csv_glob(
        (root / "diagnostics").glob("capture_*/proxy_gap_metrics.csv"),
        root / "tables" / "proxy_gap_metrics.csv",
    )
    p0.atomic_json(
        root / "diagnostics.done.json",
        {"completed": p0.utc_now(), "config_hash": cfg_hash, "units": expected, "rows": rows},
    )
    print(f"[merge] units={expected} rows={rows}", flush=True)


def cmd_summarize(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash, _exact_cfg, _exact_hash = load_configs(cfg_path)
    root = output_root(cfg)
    source = root / "tables" / "proxy_gap_metrics.csv"
    metrics = [
        "uniform_K_error_per_dim", "uniform_V_error_per_dim", "K_mean_error_fraction",
        "V_mean_error_fraction", "V_mass_weighted_error_per_dim", "V_mass_weighted_vs_uniform",
        "V_output_error_relative_energy", "V_cross_token_coherence_multiplier",
        "K_logit_error_per_pair", "K_logit_error_relative_energy",
        "K_probability_error_per_query",
        "K_output_error_relative_energy", "K_value_coupling_multiplier",
    ]
    sums: dict[tuple[int, str, str], list[float]] = {}
    layer_sums: dict[tuple[int, int, str, str], list[float]] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for metric in metrics:
                bucket = sums.setdefault((int(row["prompt_id"]), row["method"], metric), [0.0, 0.0])
                bucket[0] += float(row[metric])
                bucket[1] += 1
                layer_bucket = layer_sums.setdefault(
                    (int(row["layer"]), int(row["prompt_id"]), row["method"], metric),
                    [0.0, 0.0],
                )
                layer_bucket[0] += float(row[metric])
                layer_bucket[1] += 1
    prompt_ids = sorted({key[0] for key in sums})
    prompt_rows = []
    summary_rows = []
    for metric_index, metric in enumerate(metrics):
        changes = []
        for prompt_id in prompt_ids:
            base_total, base_n = sums[(prompt_id, "compact30", metric)]
            cand_total, cand_n = sums[(prompt_id, "s30_l10_uniform8", metric)]
            baseline = base_total / base_n
            candidate = cand_total / cand_n
            change = candidate / max(baseline, 1e-30) - 1.0
            changes.append(change)
            prompt_rows.append(
                {
                    "prompt_id": prompt_id, "metric": metric,
                    "compact30": baseline, "s30_l10_uniform8": candidate,
                    "candidate_relative_change": change,
                }
            )
        ci_low, ci_high = compact.bootstrap_ci(
            changes, int(cfg["bootstrap"]["samples"]),
            int(cfg["bootstrap"]["seed"]) + metric_index,
        )
        summary_rows.append(
            {
                "metric": metric, "prompts": len(prompt_ids),
                "mean_candidate_relative_change": float(np.mean(changes)),
                "bootstrap_95_ci_low": ci_low, "bootstrap_95_ci_high": ci_high,
                "candidate_higher_prompt_rate": float(np.mean(np.asarray(changes) > 0)),
            }
        )
    p0.write_csv(root / "tables" / "prompt_proxy_comparison.csv", prompt_rows)
    p0.write_csv(root / "tables" / "proxy_gap_summary.csv", summary_rows)
    layer_rows = []
    layers = sorted({key[0] for key in layer_sums})
    for layer in layers:
        for metric_index, metric in enumerate(metrics):
            changes = []
            for prompt_id in prompt_ids:
                base_total, base_n = layer_sums[(layer, prompt_id, "compact30", metric)]
                cand_total, cand_n = layer_sums[(layer, prompt_id, "s30_l10_uniform8", metric)]
                baseline = base_total / base_n
                candidate = cand_total / cand_n
                changes.append(candidate / max(baseline, 1e-30) - 1.0)
            ci_low, ci_high = compact.bootstrap_ci(
                changes, int(cfg["bootstrap"]["samples"]),
                int(cfg["bootstrap"]["seed"]) + 1000 + layer * 31 + metric_index,
            )
            layer_rows.append(
                {
                    "layer": layer, "metric": metric, "prompts": len(prompt_ids),
                    "mean_candidate_relative_change": float(np.mean(changes)),
                    "bootstrap_95_ci_low": ci_low, "bootstrap_95_ci_high": ci_high,
                }
            )
    p0.write_csv(root / "tables" / "layer_proxy_gap_summary.csv", layer_rows)
    by_metric = {row["metric"]: row for row in summary_rows}
    key_absolute = by_metric["K_logit_error_per_pair"]
    key = by_metric["K_logit_error_relative_energy"]
    v_uniform = by_metric["uniform_V_error_per_dim"]
    v_output = by_metric["V_output_error_relative_energy"]
    v_coherence = by_metric["V_cross_token_coherence_multiplier"]
    interpretation = {
        "schema_version": "wan5b_p1_proxy_gap_summary.v1", "completed": p0.utc_now(),
        "config_hash": cfg_hash, "status": "posthoc_explanatory_not_confirmatory",
        "prompts": len(prompt_ids),
        "K_mechanism": {
            "candidate_logit_error_relative_change": key["mean_candidate_relative_change"],
            "candidate_absolute_logit_mse_relative_change": key_absolute["mean_candidate_relative_change"],
            "interpretation": "the primary K diagnostic is normalized to reference logit energy within head/capture; raw absolute MSE is retained only to expose layer-scale confounding",
        },
        "V_mechanism": {
            "candidate_uniform_error_relative_change": v_uniform["mean_candidate_relative_change"],
            "candidate_attention_output_energy_relative_change": v_output["mean_candidate_relative_change"],
            "candidate_cross_token_coherence_relative_change": v_coherence["mean_candidate_relative_change"],
            "interpretation": "lower uniform V error can still produce higher output error when token errors add more coherently under attention",
        },
        "metric_scope": "pre-camera-output-transform decomposition; exact stored queries, keys, values and actual INT4 codebooks",
    }
    p0.atomic_json(root / "summary.json", interpretation)
    lines = [
        "# Wan2.2 5B P1 attention-proxy gap diagnostic", "",
        "This is a post-hoc explanatory analysis of the preregistered P1-S3 failure, not a new confirmatory test.", "",
        "| Metric (10xC8 vs compact30) | Mean prompt-relative change | 95% prompt bootstrap CI |",
        "|---|---:|---:|",
    ]
    for name in (
        "uniform_K_error_per_dim", "K_logit_error_relative_energy",
        "K_logit_error_per_pair", "K_output_error_relative_energy",
        "uniform_V_error_per_dim", "V_mass_weighted_error_per_dim",
        "V_cross_token_coherence_multiplier", "V_output_error_relative_energy",
    ):
        row = by_metric[name]
        lines.append(
            f"| {name} | {100*float(row['mean_candidate_relative_change']):.3f}% | "
            f"[{100*float(row['bootstrap_95_ci_low']):.3f}%, {100*float(row['bootstrap_95_ci_high']):.3f}%] |"
        )
    lines.extend([
        "", "Positive means 10xC8 is larger. V output error is decomposed before the camera output transform so the diagonal and cross-token terms are comparable.",
    ])
    report = root / "report.md"
    temporary = report.with_suffix(f".md.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, report)
    p0.atomic_json(root / "summary.done.json", {"completed": p0.utc_now(), "config_hash": cfg_hash})
    print(json.dumps(interpretation, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", required=True)
    worker.add_argument("--shard-id", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--config", required=True)
    launch.add_argument("--gpus")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--config", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--config", required=True)
    args = parser.parse_args()
    {"prepare": cmd_prepare, "worker": cmd_worker, "launch": cmd_launch, "merge": cmd_merge, "summarize": cmd_summarize}[args.command](args)


if __name__ == "__main__":
    main()
