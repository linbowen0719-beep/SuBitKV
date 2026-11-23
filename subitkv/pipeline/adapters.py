"""Adapters to the existing model implementations, without historical data paths."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from subitkv.cache_budget import rvq_schedule_budget
from subitkv.baselines import BASELINE_QUANT_TYPES

from .common import REPO, absolute, digest, identity, seal, write_json
from .workflow import prompts


def label(budget):
    if isinstance(budget, str):
        return budget
    return "avg" + f"{budget:g}".replace(".", "p")


def process(argv, env, directory):
    directory.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    merged.update({k: str(v) for k, v in env.items()})
    write_json(
        directory / "command.json",
        {"argv": [str(v) for v in argv], "env_overrides": env},
    )
    started = time.perf_counter()
    with (directory / "run.log").open("x") as log:
        result = subprocess.run(
            [str(v) for v in argv],
            cwd=REPO,
            env=merged,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(
            f"Model process failed ({result.returncode}); see {directory / 'run.log'}"
        )
    elapsed = time.perf_counter() - started
    videos = sorted(directory.rglob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(
            f"Expected one generated video in {directory}; found {len(videos)}"
        )
    return videos[0], elapsed


def quant_config(config, bundle, budget):
    if isinstance(budget, str):
        return dict(
            quant_type=BASELINE_QUANT_TYPES[budget], compression_policy="all",
            cache_position="pre_rope", k_rvq=False, v_rvq=False,
            temporal_adaptive_residual=False, num_prq_stages=1,
            cache_num_k_centroids=256, cache_num_v_centroids=256,
            quant_block_size=64, kmeans_max_iters=2,
            kivi_group_size=32, turboquant_seed=20260803,
        )
    c, m = config["calibration"], config["model_config"]
    return dict(
        compression_policy="all",
        quant_type="subitkv",
        cache_position="pre_rope",
        k_rvq=True,
        v_rvq=True,
        k_rvq_codebook_dir=str(bundle / "nonorm_k_l16c64_codebooks"),
        v_rvq_codebook_dir=str(bundle / "nonorm_v_l16c64_codebooks"),
        k_rvq_batch_size=c["assignment_batch_size"],
        v_rvq_batch_size=c["assignment_batch_size"],
        k_rvq_compute_dtype="bf16",
        v_rvq_compute_dtype="bf16",
        k_rvq_codebook_storage_dtype=c["codebook_storage_dtype"],
        v_rvq_codebook_storage_dtype=c["codebook_storage_dtype"],
        k_rvq_scale_storage_dtype=c["scale_storage_dtype"],
        v_rvq_scale_storage_dtype=c["scale_storage_dtype"],
        rvq_relative_stop=True,
        rvq_relative_stop_threshold=c["early_stop"]["threshold"],
        rvq_relative_stop_min_depth=c["early_stop"]["min_depth"],
        rvq_relative_stop_max_depth=c["early_stop"]["max_depth"],
        rvq_residual_quantize=True,
        rvq_k_residual_codec="frame_mean_gaussian",
        rvq_v_residual_codec="progressive_sign_adaptive_absmax_scalar",
        routed_gaussian_bits=8,
        routed_gaussian_group_size=c["residual"]["group_size"],
        routed_gaussian_hadamard_group_size=c["residual"]["group_size"],
        routed_gaussian_scale_refine_iters=1,
        routed_gaussian_seed=c["residual"]["codec_seed"],
        routed_gaussian_routing_path=str(bundle / "routing.json"),
        routed_gaussian_tree_params_path=str(bundle / "tree_params.pt"),
        progressive_v_absmax_bits=8,
        progressive_v_absmax_group_size=c["residual"]["group_size"],
        progressive_v_absmax_method="midpoint_msb",
        progressive_v_absmax_params_path=str(bundle / "v_params.pt"),
        frame_mean_spatial_tokens=m["spatial_tokens"],
        temporal_adaptive_residual=True,
        temporal_adaptive_profile_path=str(
            bundle / "profiles" / (label(budget) + ".json")
        ),
        temporal_adaptive_capacity_bit_chunks=max(
            rvq_schedule_budget(
                config, c["schedule"]["budgets"][f"{budget:g}"]["terminal"]
            ).capacities
        ),
        temporal_adaptive_allocation_mode="frozen_joint_schedule",
    )


def generate(
    run,
    prompt_id,
    prompt,
    split,
    destination,
    *,
    budget=None,
    capture=None,
    extra_env=None,
):
    c, m = run.config, run.config["model_config"]
    conditioning_split = "calibration" if split == "train" and c["profile"] in {"full", "validation"} else split
    gpu = c["runtime"]["gpus"][prompt_id % len(c["runtime"]["gpus"])]
    destination.mkdir(parents=True, exist_ok=True)
    conditioning_hash = None
    if c["model"] == "hyworldplay_8b":
        image_path = Path(c["data"]["single_conditioning_image"]) if split == "custom" else (
            absolute(c["data"]["conditioning_images"]) / conditioning_split / f"{prompt_id:04d}.png"
        )
        conditioning_hash = digest(image_path)
    frozen_input = run.root / "inputs" / split / f"{prompt_id:04d}.json"
    input_record = dict(
        prompt=prompt, conditioning_sha256=conditioning_hash, seed=m["seed"]
    )
    if frozen_input.exists():
        if json.loads(frozen_input.read_text()) != input_record:
            raise ValueError(f"Input changed within run: {frozen_input}")
    else:
        write_json(frozen_input, input_record)
    prompt_file = destination / "prompt.txt"
    prompt_file.write_text(prompt + "\n")
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": str(REPO),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    from subitkv.models import hyworldplay, self_forcing, wan

    adapter = {
        "wan2_2_5b": wan,
        "self_forcing": self_forcing,
        "hyworldplay_8b": hyworldplay,
    }[c["model"]]
    argv, env = adapter.build_command(
        run,
        prompt_id,
        prompt,
        conditioning_split,
        destination,
        prompt_file,
        env,
        budget=budget,
        capture=capture,
    )
    env.update(extra_env or {})
    video, elapsed = process(argv, env, destination)
    if capture and c["model"] == "wan2_2_5b":
        from . import capture as diag

        diag.merge_prompt_spans(
            capture,
            "train",
            prompt_id,
            prompt,
            prompt_id,
            list(range(m["layers"])),
            m["spatial_tokens"],
            False,
            "V",
        )
        # Keep a bounded, deterministic set of complete latent frames for
        # codebook fitting. Attention weights are calibrated by a separate
        # teacher pass, so this does not approximate its softmax denominator.
        import torch

        for layer in range(m["layers"]):
            path = capture / "kv_dump/train" / str(prompt_id) / f"layer_{layer}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            total_frames = payload["K"].shape[0]
            take = min(total_frames, m["capture_tokens"] // m["spatial_tokens"])
            generator = torch.Generator().manual_seed(
                c["calibration"]["seed"] + prompt_id * 1000003 + layer
            )
            indices = (
                torch.randperm(total_frames, generator=generator)[:take].sort().values
            )
            for name in ("Q", "K", "V"):
                payload[name] = payload[name].index_select(0, indices).contiguous()
            payload["metadata"].update(
                sampled_frame_indices=indices.tolist(),
                source_frames=total_frames,
                T=take,
            )
            temporary = path.with_suffix(".sampled.pt")
            torch.save(payload, temporary)
            temporary.replace(path)
    image_hash = None
    if c["model"] == "hyworldplay_8b":
        image_hash = conditioning_hash
    runtime = runtime_metrics(c, destination)
    runtime.update(
        process_seconds=elapsed,
        timing_scope="subprocess launch through exit, including load and decode",
    )
    record = dict(
        method="bf16" if budget is None else label(budget),
        prompt_id=prompt_id,
        prompt=prompt,
        seed=m["seed"],
        input_hash=identity([prompt, image_hash]),
        video=str(video),
        video_sha256=digest(video),
        chunk_boundaries=list(
            range(m["first_chunk_frames"], m["frames"], m["chunk_frames"])
        ),
        runtime_metrics=runtime,
    )
    dataset_set = "calibration" if split == "train" else c["workload"].get("eval_set")
    if dataset_set:
        from .dataset import verify_splits

        record.update(dataset_set=dataset_set, dataset_row=verify_splits(REPO)["splits"][dataset_set]["dataset_rows"][prompt_id])
    write_json(destination / "record.json", record)
    return record


def runtime_metrics(config, directory):
    """Retain raw accounting alongside comparable physical K/V payload columns."""
    raw = {}
    if config["model"] == "wan2_2_5b":
        path = directory / "metrics.csv"
        if path.exists():
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) == 1:
                raw = rows[0]
    elif (directory / "runtime.json").exists():
        raw = json.loads((directory / "runtime.json").read_text())
    if not raw:
        raise RuntimeError(f"Model did not produce runtime accounting: {directory}")

    def numeric(*keys):
        for key in keys:
            value = raw.get(key)
            if value not in (None, "", "None"):
                return float(value)
        return None

    result = dict(
        k_payload_mib=numeric("k_payload_mib"),
        v_payload_mib=numeric("v_payload_mib"),
        kv_payload_mib=numeric(
            "kv_payload_mib", "kv_cache_mem_mb", "kv_cache_total_mib"
        ),
        kv_peak_payload_mib=numeric(
            "kv_peak_payload_mib", "kv_cache_compressed_vision_peak_mib"
        ),
        shared_codebook_mib=numeric("shared_codebook_mib"),
        codec_workspace_mib=numeric("codec_workspace_mib"),
        codec_workspace_calls=numeric("codec_workspace_calls"),
        shared_parameters_mib=max(numeric("shared_codebook_mib") or 0, numeric("codec_resident_shared_mib") or 0),
        peak_gpu_allocated_mib=max(numeric("peak_gpu_mem_mb", "peak_allocated_mib") or 0, numeric("codec_process_peak_mib") or 0),
        memory_accounting_scope=raw.get("memory_accounting_scope"),
        payload_includes_metadata=True,
        generation_seconds=numeric(
            "gen_time_s", "e2e_time_s", "e2e_seconds", "mp4_complete_elapsed_seconds"
        ),
        raw_accounting_path=str(
            directory
            / ("metrics.csv" if config["model"] == "wan2_2_5b" else "runtime.json")
        ),
        memory_units="MiB",
        baseline="same-input BF16 K/V payload",
    )
    if config["model"] == "hyworldplay_8b":
        # Use runtime's terminal snapshot for separate K/V accounting.
        from subitkv.models import accounting as helper

        summary = helper.summarize_avg2_accounting(
            raw.get("kv_codec_encode_events", []), None
        )
        if summary:
            terminal = summary["terminal_snapshot"]
            result.update(
                k_payload_mib=(
                    terminal["physical_local_k_bytes"]
                )
                / 1024**2,
                v_payload_mib=(
                    terminal["physical_local_v_bytes"]
                )
                / 1024**2,
                bf16_kv_payload_mib=terminal["bf16_kv_bytes"] / 1024**2,
                physical_residual_bpp=terminal["physical_residual_bpp"],
                nominal_residual_avg_bits=terminal["nominal_residual_avg_bits"],
            )
            tables = (terminal["physical_shared_k_bytes"] + terminal["physical_shared_v_bytes"]) / 1024**2
            result["kv_payload_mib"] = terminal["physical_payload_bytes"] / 1024**2 - tables
            result["shared_parameters_mib"] = max((result["shared_codebook_mib"] or 0) + tables, result["shared_parameters_mib"])
        if result["kv_payload_mib"] is None and raw.get("kv_cache_codec") == "none":
            value = numeric("kv_cache_bf16_vision_peak_mib")
            result.update(
                kv_payload_mib=value,
                k_payload_mib=value / 2 if value else None,
                v_payload_mib=value / 2 if value else None,
            )
        result["generation_timing_scope"] = (
            "HY generate_video entry through completed video encoding"
        )
    else:
        result["generation_timing_scope"] = (
            "native model runtime; see raw_accounting_path"
        )
    from subitkv.memory import cache_footprint

    result["kv_total_mib"] = cache_footprint(result["kv_payload_mib"], result["shared_parameters_mib"], result["codec_workspace_mib"])
    return result


def attempt(parent):
    number = 1
    while (parent / f"attempt_{number:02d}").exists():
        number += 1
    return parent / f"attempt_{number:02d}"


def collect(run):
    config = run.config
    values = prompts(config, "train")
    run.capture.mkdir(parents=True, exist_ok=True)
    split_path = run.capture / "prompt_split.json"
    if not split_path.exists():
        write_json(
            split_path,
            dict(
                seed=config["calibration"]["seed"],
                train=[
                    dict(prompt_id=i, source_index=i, prompt=p)
                    for i, p in enumerate(values)
                ],
                test=[],
            ),
        )
    for i, prompt in enumerate(values):

        def collect_one(i=i, prompt=prompt):
            directory = attempt(run.root / "samples" / f"prompt_{i:04d}")
            local_capture = directory / "capture"
            generate(run, i, prompt, "train", directory, capture=local_capture)
            target = run.capture / "kv_dump/train" / str(i)
            source = local_capture / "kv_dump/train" / str(i)
            paths = [
                source / f"layer_{layer}.pt"
                for layer in range(config["model_config"]["layers"])
            ]
            if not all(p.is_file() for p in paths):
                raise RuntimeError(f"Incomplete KV capture: {source}")
            # Materialize files so published captures never depend on external symlinks.
            target.mkdir(parents=True, exist_ok=True)
            for path in paths:
                destination = target / path.name
                if destination.exists():
                    if digest(destination) != digest(path):
                        raise ValueError(f"Partial capture differs: {destination}")
                else:
                    shutil.copy2(path, destination)
            return list(target.glob("*.pt")) + [directory / "record.json"]

        run.task(f"collect_{i}", collect_one)

    def finish():
        seal(run.capture, config["model"], identity(config), split="train")
        return [run.capture / "manifest.json"]

    run.task("collect_manifest", finish)


def infer(run, budget=None, reference_only=False, baseline=None):
    values = prompts(run.config, "eval")
    budgets = (
        []
        if reference_only
        else ([baseline] if baseline else [budget] if budget is not None else run.config["budgets"])
    )
    for selected in [None, *budgets]:
        method = "bf16" if selected is None else label(selected)
        for i, prompt in enumerate(values):

            def inference_one(i=i, prompt=prompt, selected=selected, method=method):
                directory = attempt(run.root / "videos" / method / f"prompt_{i:04d}")
                record = generate(run, i, prompt, run.config["workload"].get("eval_set", "eval"), directory, budget=selected)
                return [directory / "record.json", Path(record["video"])]

            run.task(f"infer_{method}_{i}", inference_one)
    records = []
    for marker in sorted((run.root / "status").glob("infer_*.json")):
        data = json.loads(marker.read_text())
        for name, expected in data["files"].items():
            if digest(run.root / name) != expected:
                raise ValueError(f"Inference artifact changed: {name}")
            if name.endswith("/record.json"):
                records.append(json.loads((run.root / name).read_text()))
    # This is an index only; immutable per-task records remain the source of truth.
    temporary = run.root / "videos.json.tmp"
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    temporary.replace(run.root / "videos.json")
