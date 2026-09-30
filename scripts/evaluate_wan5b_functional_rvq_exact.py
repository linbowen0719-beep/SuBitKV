#!/usr/bin/env python3
"""Exact-attention evaluation for independently switchable functional RVQ ideas.

This replays the saved pre-position historical K/V through the exact standard
RoPE and camera PRoPE paths.  Only historical tokens are reconstructed; the
current segment remains bit-identical BF16.  All variants use the same 30-bit
logical index budget and the actual INT4-dequantized runtime codebooks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_p1_exact_replay as exact  # noqa: E402
import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402
import train_wan5b_functional_rvq as trainer  # noqa: E402
from subitkv import rvq  # noqa: E402


PRIMARY_COMPARISONS = {
    "qaware_k": "K_only",
    "vaware_v": "V_only",
    "qaware_vaware": "K_and_V",
    "kv_independent": "K_and_V",
    "combined": "K_and_V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "worker", "launch", "merge", "summarize", "all"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--capture-config", type=Path)
        child.add_argument("--output-name", default="")
        child.add_argument("--capture-label", default="")
        if command == "worker":
            child.add_argument("--shard-id", type=int, required=True)
            child.add_argument("--num-shards", type=int, required=True)
        if command in {"launch", "all"}:
            child.add_argument("--gpus", default="")
    return parser.parse_args()


def load(path: Path) -> tuple[dict[str, Any], str, Path]:
    resolved = p0.resolve_path(path)
    cfg, cfg_hash = p0.parse_config(resolved)
    return cfg, cfg_hash, resolved


def root(cfg: dict[str, Any]) -> Path:
    name = os.getenv("WAN5B_FUNCTIONAL_EXACT_OUTPUT_NAME", "exact_evaluation")
    if not name or Path(name).name != name:
        raise ValueError(f"Unsafe exact output name: {name!r}")
    return trainer.output_root(cfg) / name


def capture_tasks(cfg: dict[str, Any]) -> list[tuple[int, Path]]:
    override = os.getenv("WAN5B_FUNCTIONAL_EXACT_CAPTURE_CONFIG", "").strip()
    capture_cfg, _capture_hash = p0.parse_config(
        p0.resolve_path(override or cfg["exact_evaluation"]["capture_config"])
    )
    return [
        (index, p0.resolve_path(value))
        for index, value in enumerate(capture_cfg["h6"]["capture_files"])
    ]


def unit_paths(cfg: dict[str, Any], index: int) -> dict[str, Path]:
    directory = root(cfg) / f"capture_{index:02d}"
    return {
        "metrics": directory / "functional_metrics.csv",
        "validation": directory / "validation.json",
        "done": directory / "done.json",
    }


def runtime_levels(
    cfg: dict[str, Any], seed: int, variant: str, tensor_name: str,
    layer: int, device: torch.device,
) -> list[torch.Tensor]:
    directory = (
        trainer.output_root(cfg) / "runtime" / f"seed_{seed}" / variant
        / tensor_name.lower()
    )
    payload = rvq.load_layer_codebooks_from_dir(
        directory,
        layer_idx=layer,
        device=device,
        compute_dtype=torch.float32,
        codebook_storage_dtype="int4",
        num_heads=p0.HEADS,
    )
    if not rvq._is_variable_cardinality_rvq(payload):
        raise ValueError(f"Expected variable-cardinality runtime codebooks: {directory}")
    return [
        rvq._codebook_level(payload, level, torch.float32).float().contiguous()
        for level in range(len(rvq._variable_cardinalities(payload)))
    ]


def capture_prompt(capture_path: Path) -> tuple[int, str]:
    info_path = capture_path.parents[3] / "run_info.json"
    if not info_path.exists():
        return -1, "unknown_capture_prompt"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    prompt_id = int(info.get("prompt_id", info.get("prompt_start", -1)))
    prompts = info.get("prompt_list", [])
    prompt = str(prompts[0]) if prompts else "unknown_capture_prompt"
    return prompt_id, prompt


def run_capture(
    cfg: dict[str, Any], cfg_hash: str, index: int, capture_path: Path,
    device: torch.device,
) -> None:
    paths = unit_paths(cfg, index)
    if all(path.exists() for path in paths.values()):
        return
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite incomplete exact unit: {existing}")
    started = time.time()
    raw = torch.load(capture_path, map_location="cpu", weights_only=False)
    metadata = exact.validate_capture(raw)
    layer = int(metadata["layer"])
    (
        tensors, evidence, validation, query_output_matrix, _full_kv,
        query_positions, key_positions,
    ) = exact.positional_replay(raw, device)
    history_tokens = int(metadata["cache_tokens"])
    key_tokens = int(tensors["key_pre"].shape[2])
    prompt_id, prompt = capture_prompt(capture_path)
    rows: list[dict[str, Any]] = []

    for seed_value in cfg["initialization_seeds"]:
        seed = int(seed_value)
        histories: dict[tuple[str, str], torch.Tensor] = {}
        for variant in trainer.VARIANTS:
            for tensor_name, reference_key in (("K", "key_pre"), ("V", "value_pre")):
                cache_key = (variant, tensor_name)
                levels = runtime_levels(cfg, seed, variant, tensor_name, layer, device)
                histories[cache_key] = exact.reconstruct_history(
                    tensors[reference_key][:, :, :history_tokens],
                    levels,
                    device,
                    assignment_batch_size=4096,
                )

        for variant in trainer.VARIANTS:
            k_history = histories[(variant, "K")]
            v_history = histories[(variant, "V")]
            standard_k, standard_v, prope_k, prope_v, _history_kv = (
                exact.candidate_positional_tensors(
                    tensors, evidence, history_tokens, k_history, v_history
                )
            )
            current_errors = {
                "standard_k": p0.max_abs_error(
                    standard_k[:, :, history_tokens:],
                    tensors["key_post_rope"][:, :, history_tokens:],
                ),
                "standard_v": p0.max_abs_error(
                    standard_v[:, :, history_tokens:],
                    tensors["value_post_rope"][:, :, history_tokens:],
                ),
                "prope_k": p0.max_abs_error(
                    prope_k[:, :, history_tokens:],
                    tensors["key_post_prope"][:, :, history_tokens:],
                ),
                "prope_v": p0.max_abs_error(
                    prope_v[:, :, history_tokens:],
                    tensors["value_post_prope"][:, :, history_tokens:],
                ),
            }
            if any(value != 0 for value in current_errors.values()):
                raise RuntimeError(f"Current BF16 segment changed: {current_errors}")
            halves = {
                "standard_rope": (
                    tensors["query_post_rope"], tensors["key_post_rope"],
                    tensors["value_post_rope"], standard_k, standard_v, None,
                ),
                "camera_prope": (
                    tensors["query_post_prope"], tensors["key_post_prope"],
                    tensors["value_post_prope"], prope_k, prope_v, query_output_matrix,
                ),
            }
            k_rel = [
                p0.relative_mse(
                    k_history[:, head], tensors["key_pre"][:, head, :history_tokens]
                )
                for head in range(p0.HEADS)
            ]
            v_rel = [
                p0.relative_mse(
                    v_history[:, head], tensors["value_pre"][:, head, :history_tokens]
                )
                for head in range(p0.HEADS)
            ]
            for half_name, values in halves.items():
                query, key_ref, value_ref, key_hat, value_hat, output_matrix = values
                for head in range(p0.HEADS):
                    for intervention, candidate_key, candidate_value in (
                        ("K_only", key_hat[:, head : head + 1], value_ref[:, head : head + 1]),
                        ("V_only", key_ref[:, head : head + 1], value_hat[:, head : head + 1]),
                        ("K_and_V", key_hat[:, head : head + 1], value_hat[:, head : head + 1]),
                    ):
                        metrics = p0.head_attention_metrics(
                            query=query[:, head : head + 1],
                            key_reference=key_ref[:, head : head + 1],
                            key_candidate=candidate_key,
                            value_reference=value_ref[:, head : head + 1],
                            value_candidate=candidate_value,
                            history_tokens=history_tokens,
                            query_positions=query_positions,
                            key_positions=key_positions,
                            query_batch_size=int(cfg["exact_evaluation"]["query_batch_size"]),
                            topk=int(cfg["exact_evaluation"]["topk"]),
                            output_matrix=output_matrix,
                            evidence=evidence,
                        )
                        rows.append(
                            {
                                "study_id": cfg["study_id"],
                                "config_hash": cfg_hash,
                                "prompt_id": prompt_id,
                                "prompt": prompt,
                                "capture_index": index,
                                "capture_path": str(capture_path),
                                "initialization_seed": seed,
                                "layer": layer,
                                "head": head,
                                "variant": variant,
                                "logical_index_bits_per_vector": 30,
                                "physical_index_bytes_per_vector": 4,
                                "runtime_codebook_storage": "int4_per_centroid_fp16_scale",
                                "attention_half": half_name,
                                "intervention": intervention,
                                "hook_stage": "pre_positional_historical_only",
                                "history_tokens": history_tokens,
                                "current_bf16_tokens": key_tokens - history_tokens,
                                "historical_K_relative_mse": k_rel[head],
                                "historical_V_relative_mse": v_rel[head],
                                **metrics,
                            }
                        )

    validation.update(
        {
            "capture_path": str(capture_path),
            "capture_sha256": p0.sha256_file(capture_path),
            "current_segment_untouched": True,
            "runtime_codebooks": "INT4-dequantized production directories",
            "logical_index_bits_per_vector": 30,
            "physical_index_bytes_per_vector": 4,
        }
    )
    p0.write_csv(paths["metrics"], rows)
    p0.atomic_json(paths["validation"], validation)
    p0.atomic_json(
        paths["done"],
        {
            "completed": p0.utc_now(),
            "config_hash": cfg_hash,
            "capture_index": index,
            "layer": layer,
            "rows": len(rows),
            "seconds": time.time() - started,
        },
    )
    print(
        f"[exact] capture={index} prompt={prompt_id} layer={layer} "
        f"rows={len(rows)} seconds={time.time()-started:.1f}",
        flush=True,
    )


def prepare(cfg: dict[str, Any], cfg_hash: str, cfg_path: Path) -> None:
    destination = root(cfg)
    info_path = destination / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("config_hash") != cfg_hash:
            raise ValueError(f"Config hash mismatch in {info_path}")
        print(f"[prepare-skip] {info_path}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    runtime_manifest = trainer.output_root(cfg) / "runtime" / "manifest.json"
    capture_cfg_path = p0.resolve_path(
        os.getenv("WAN5B_FUNCTIONAL_EXACT_CAPTURE_CONFIG", "").strip()
        or cfg["exact_evaluation"]["capture_config"]
    )
    tasks = capture_tasks(cfg)
    prompts = [capture_prompt(path)[1] for _index, path in tasks]
    p0.atomic_json(
        info_path,
        {
            "git_commit": p0.git_commit(),
            "config_path": str(cfg_path.resolve()),
            "config_hash": cfg_hash,
            "codebook_path": str(runtime_manifest.resolve()),
            "codebook_hash": p0.sha256_file(runtime_manifest),
            "hostname": p0.socket.gethostname(),
            "gpu_type": p0.gpu_name(),
            "seed": int(cfg["seed"]),
            "prompt_list": sorted(set(prompts)),
            "timestamp": p0.utc_now(),
            "capture_files": [
                {"path": str(path), "sha256": p0.sha256_file(path)}
                for _index, path in tasks
            ],
            "capture_config_path": str(capture_cfg_path.resolve()),
            "capture_config_hash": p0.sha256_file(capture_cfg_path),
            "capture_label": os.getenv(
                "WAN5B_FUNCTIONAL_EXACT_CAPTURE_LABEL", "all_layer_discovery_overlap"
            ),
            "statistical_unit": "prompt",
            "current_segment_policy": "BF16 and bit-exactly untouched",
        },
    )


def worker(
    cfg: dict[str, Any], cfg_hash: str, shard_id: int, num_shards: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    selected = [
        task for task_index, task in enumerate(capture_tasks(cfg))
        if task_index % int(num_shards) == int(shard_id)
    ]
    print(
        f"[worker] shard={shard_id}/{num_shards} tasks={len(selected)} "
        f"gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    for index, path in selected:
        run_capture(cfg, cfg_hash, index, path, torch.device("cuda:0"))


def launch(cfg: dict[str, Any], cfg_path: Path, raw_gpus: str) -> None:
    gpu_ids = [int(value) for value in (raw_gpus.split(",") if raw_gpus else cfg["gpus"])]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError(f"Expected four distinct GPUs, got {gpu_ids}")
    log_dir = root(cfg) / "logs" / f"exact_{p0.utc_now().replace(':', '').replace('+', '_')}"
    log_dir.mkdir(parents=True, exist_ok=False)
    processes = []
    for shard, gpu in enumerate(gpu_ids):
        log_path = log_dir / f"gpu_{gpu}_shard_{shard}.log"
        handle = log_path.open("x", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            str(cfg["python"]), str(Path(__file__).resolve()), "worker",
            "--config", str(cfg_path), "--shard-id", str(shard),
            "--num-shards", str(len(gpu_ids)),
        ]
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        processes.append((gpu, process, handle, log_path))
        print(f"[launch] gpu={gpu} pid={process.pid} log={log_path}", flush=True)
    last_report = 0.0
    try:
        while any(process.poll() is None for _gpu, process, _handle, _path in processes):
            if time.time() - last_report >= 15:
                print(
                    "[monitor] " + ", ".join(
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
    failures = [item for item in processes if item[1].returncode != 0]
    if failures:
        for gpu, process, _handle, path in failures:
            print(
                f"[failed] gpu={gpu} rc={process.returncode} log={path}\n"
                f"{path.read_text(encoding='utf-8')[-12000:]}",
                flush=True,
            )
        raise RuntimeError(f"{len(failures)} exact workers failed")


def merge(cfg: dict[str, Any], cfg_hash: str) -> None:
    expected = len(capture_tasks(cfg))
    done = sorted(root(cfg).glob("capture_*/done.json"))
    if len(done) != expected:
        raise RuntimeError(f"Incomplete exact replay: {len(done)}/{expected}")
    count = p0.merge_csv_glob(
        root(cfg).glob("capture_*/functional_metrics.csv"),
        root(cfg) / "tables" / "functional_metrics.csv",
    )
    p0.atomic_json(
        root(cfg) / "merge.done.json",
        {"completed": p0.utc_now(), "config_hash": cfg_hash, "units": expected, "rows": count},
    )
    print(f"[merge] units={expected} rows={count}", flush=True)


def bootstrap_ci(values: list[float], seed: int, samples: int = 10000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(cfg: dict[str, Any], cfg_hash: str) -> None:
    path = root(cfg) / "tables" / "functional_metrics.csv"
    aggregates: dict[tuple[int, int, str, str], list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["prompt_id"]), int(row["initialization_seed"]),
                row["variant"], row["intervention"],
            )
            aggregates.setdefault(key, []).append(float(row["attention_output_relative_l2"]))
    means = {key: float(np.mean(values)) for key, values in aggregates.items()}
    prompt_ids = sorted({key[0] for key in means})
    seeds = sorted({key[1] for key in means})
    rows = []
    for variant, intervention in PRIMARY_COMPARISONS.items():
        prompt_improvements = []
        seed_unit_improvements = []
        for prompt_id in prompt_ids:
            baseline = float(np.mean([
                means[(prompt_id, seed, "fullmap_baseline", intervention)] for seed in seeds
            ]))
            candidate = float(np.mean([
                means[(prompt_id, seed, variant, intervention)] for seed in seeds
            ]))
            prompt_improvements.append((baseline - candidate) / max(baseline, 1e-20))
            for seed in seeds:
                base_seed = means[(prompt_id, seed, "fullmap_baseline", intervention)]
                candidate_seed = means[(prompt_id, seed, variant, intervention)]
                seed_unit_improvements.append(
                    (base_seed - candidate_seed) / max(base_seed, 1e-20)
                )
        low, high = bootstrap_ci(prompt_improvements, int(cfg["seed"]) + len(rows) * 1009)
        allocation_overlap = (
            variant in {"kv_independent", "combined"}
            and os.getenv(
                "WAN5B_FUNCTIONAL_EXACT_CAPTURE_LABEL", "all_layer_discovery_overlap"
            ) == "all_layer_discovery_overlap"
        )
        rows.append(
            {
                "variant": variant,
                "baseline": "fullmap_baseline",
                "primary_intervention": intervention,
                "prompts": len(prompt_improvements),
                "initialization_seeds": len(seeds),
                "mean_prompt_relative_improvement": float(np.mean(prompt_improvements)),
                "median_prompt_relative_improvement": float(np.median(prompt_improvements)),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "prompt_win_rate": float(np.mean(np.asarray(prompt_improvements) > 0)),
                "prompt_seed_unit_win_rate": float(
                    np.mean(np.asarray(seed_unit_improvements) > 0)
                ),
                "status": (
                    "exploratory_discovery_prompt_overlap"
                    if allocation_overlap else "external_prompt_confirmation"
                ),
            }
        )
    p0.write_csv(root(cfg) / "tables" / "primary_comparisons.csv", rows)
    validations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root(cfg).glob("capture_*/validation.json"))
    ]
    summary = {
        "schema_version": "wan5b_functional_rvq_exact.v1",
        "completed": p0.utc_now(),
        "config_hash": cfg_hash,
        "prompt_ids": prompt_ids,
        "initialization_seeds": seeds,
        "current_segment_untouched_all": all(
            value.get("current_segment_untouched") is True for value in validations
        ),
        "exact_transform_replay_all": all(
            value.get("exact_transform_replay") is True for value in validations
        ),
        "comparisons": rows,
    }
    p0.atomic_json(root(cfg) / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.capture_config is not None:
        os.environ["WAN5B_FUNCTIONAL_EXACT_CAPTURE_CONFIG"] = str(
            p0.resolve_path(args.capture_config)
        )
    if args.output_name:
        os.environ["WAN5B_FUNCTIONAL_EXACT_OUTPUT_NAME"] = str(args.output_name)
    if args.capture_label:
        os.environ["WAN5B_FUNCTIONAL_EXACT_CAPTURE_LABEL"] = str(args.capture_label)
    cfg, cfg_hash, cfg_path = load(args.config)
    if args.command in {"prepare", "all"}:
        prepare(cfg, cfg_hash, cfg_path)
    if args.command == "worker":
        worker(cfg, cfg_hash, args.shard_id, args.num_shards)
    if args.command in {"launch", "all"}:
        launch(cfg, cfg_path, args.gpus)
    if args.command in {"merge", "all"}:
        merge(cfg, cfg_hash)
    if args.command in {"summarize", "all"}:
        summarize(cfg, cfg_hash)


if __name__ == "__main__":
    main()
