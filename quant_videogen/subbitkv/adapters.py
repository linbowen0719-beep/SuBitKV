"""Adapters to the existing model implementations, without historical data paths."""

from __future__ import annotations

import json
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from quant_videogen.cache_budget import rvq_schedule_budget

from .common import REPO, absolute, digest, identity, load_module, seal, write_json
from .workflow import prompts


def label(budget):
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
    c, m = config["calibration"], config["model_config"]
    return dict(
        compression_policy="all",
        quant_type="triton-nstages-kmeans-int4",
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
    gpu = c["runtime"]["gpus"][prompt_id % len(c["runtime"]["gpus"])]
    destination.mkdir(parents=True, exist_ok=True)
    conditioning_hash = None
    if c["model"] == "hyworldplay_8b":
        image_path = (
            absolute(c["data"]["conditioning_images"]) / split / f"{prompt_id:04d}.png"
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
    if c["model"] == "wan2_2_5b":
        argv = [
            sys.executable,
            REPO / "scripts/eval/run_small_eval.py",
            "--prompt-file",
            prompt_file,
            "--prompt-start",
            0,
            "--num-prompts",
            1,
            "--output-dir",
            destination,
            "--seed",
            m["seed"],
            "--image_path",
            "none",
            "--model_id",
            run.models / m["base"],
            "--ar_model_path",
            run.models / m["transformer"],
            "--ckpt_path",
            run.models / m["checkpoint"],
            "--offload_text_encoder",
            "--kv_logical_head_dim",
            256,
            "--kv_quant_factor",
            1,
        ]
        for key, field in (
            ("num_frames", "frames"),
            ("num_chunk", "chunks"),
            ("num_inference_steps", "steps"),
            ("memory_frames", "memory_frames"),
            ("temporal_context_size", "temporal_context_size"),
            ("pred_latent_size", "pred_latent_size"),
            ("pose", "pose"),
        ):
            argv.extend(["--" + key, m[field]])
        if budget is None:
            argv += ["--setting", "bf16-prerope"]
        else:
            q = quant_config(c, run.bundle, budget)
            argv += [
                "--setting",
                "kvrvq-i4",
                "--rvq_codebook_storage_dtype",
                c["calibration"]["codebook_storage_dtype"],
                "--rvq_scale_storage_dtype",
                c["calibration"]["scale_storage_dtype"],
                "--rvq_num_levels",
                c["calibration"]["levels"],
            ]
            keep = (
                "rvq_",
                "routed_gaussian_",
                "progressive_v_absmax_",
                "temporal_adaptive_",
            )
            for key, value in q.items():
                if key == "temporal_adaptive_residual":
                    continue
                if key.startswith(keep) or key in (
                    "k_rvq_codebook_dir",
                    "v_rvq_codebook_dir",
                    "k_rvq_batch_size",
                    "v_rvq_batch_size",
                ):
                    argv.extend(["--" + key] if value is True else ["--" + key, value])
            argv += ["--kv_quant_temporal_adaptive_residual"]
        env["HYWORLDPLAY_KV_CACHE_MODE"] = "pre_kv_reconstruct"
        if budget is None:
            env["HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD"] = "1"
        if capture:
            env.update(
                HYWORLDPLAY_RVQ_DIAG_DUMP="1",
                HYWORLDPLAY_RVQ_DIAG_DUMP_DIR=str(capture),
                HYWORLDPLAY_RVQ_DIAG_LAYERS=",".join(map(str, range(m["layers"]))),
                HYWORLDPLAY_RVQ_DIAG_POST_CHUNK_CACHE="1",
                HYWORLDPLAY_RVQ_DIAG_SPLIT="train",
                HYWORLDPLAY_RVQ_DIAG_PROMPT_ID=str(prompt_id),
                HYWORLDPLAY_RVQ_DIAG_PROMPT=prompt,
            )
    elif c["model"] == "self_forcing":
        base = yaml.safe_load(
            (
                REPO / "experiments/Self-Forcing/configs/self_forcing_dmd.yaml"
            ).read_text()
        )
        base.update(
            denoising_step_list=m["steps"],
            guidance_scale=m["guidance_scale"],
            seed=m["seed"],
        )
        config_path = destination / "model.yaml"
        config_path.write_text(yaml.safe_dump(base))
        q = (
            quant_config(c, run.bundle, budget)
            if budget is not None
            else dict(
                quant_type="none",
                compression_policy="all",
                cache_position="pre_rope",
                kv_cache_cpu_offload=True,
            )
        )
        if capture:
            q.update(
                calibration_dump_dir=str(capture),
                calibration_prompt_index=prompt_id,
                calibration_sample_tokens=m["capture_tokens"],
                calibration_seed=c["calibration"]["seed"],
                calibration_sampling_mode="contiguous_generation_chunk",
            )
        quant_path = destination / "quant.json"
        write_json(quant_path, q)
        argv = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--standalone",
            REPO / "experiments/Self-Forcing/inference.py",
            "--config_path",
            config_path,
            "--checkpoint_path",
            run.models / m["checkpoint"],
            "--data_path",
            prompt_file,
            "--output_folder",
            destination,
            "--num_samples",
            1,
            "--num_output_frames",
            m["latent_frames"],
            "--local_attn_size",
            m["latent_frames"],
            "--seed",
            m["seed"],
            "--use_ema",
            "--save_with_index",
            "--quant_config_json",
            quant_path,
        ]
        env.update(
            QVG_CKPT_ROOT=str(run.models),
            SELF_FORCING_METRICS_PATH=str(destination / "runtime.json"),
            PYTHONPATH=f"{REPO / 'experiments/Self-Forcing'}:{REPO}",
        )
    else:
        image = (
            absolute(c["data"]["conditioning_images"]) / split / f"{prompt_id:04d}.png"
        )
        if not image.is_file():
            raise FileNotFoundError(f"HY-WorldPlay I2V input required: {image}")
        env.update(
            PYTHON=sys.executable,
            MODEL_PATH=str(run.models / m["base"]),
            ACTION_CKPT=str(run.models / m["checkpoint"]),
            IMAGE_PATH=str(image),
            PROMPT=prompt,
            OUTPUT_PATH=str(destination),
            VIDEO_LENGTH=str(m["frames"]),
            SEED=str(m["seed"]),
            NUM_INFERENCE_STEPS=str(m["steps"]),
            WIDTH=str(m["width"]),
            HEIGHT=str(m["height"]),
            POSE=m["pose"],
            FULL_HISTORY_ATTENTION="true",
            APPEND_ONLY_VISION_CACHE="true",
            KV_CACHE_OFFLOAD="true",
            KV_CODEC_FRAME_SPATIAL_TOKENS=str(m["spatial_tokens"]),
            TERMINAL_CACHE_AUDIT="true" if budget is not None else "false",
            KV_CAPTURE_HISTORY_CALLS=str(c["calibration"]["schedule"]["horizon"]),
        )
        if budget is not None:
            runtime_settings = destination / "runtime_settings.json"
            write_json(runtime_settings, c)
            env["SUBBITKV_RUNTIME_SETTINGS"] = str(runtime_settings)
            suffix = label(budget).replace("avg0p2", "avg0p2_alt")
            env.update(
                KV_CACHE_CODEC="rvq_ours_" + suffix,
                KV_CODEC_K_CODEBOOK_DIR=str(run.bundle / "nonorm_k_l16c64_codebooks"),
                KV_CODEC_V_CODEBOOK_DIR=str(run.bundle / "nonorm_v_l16c64_codebooks"),
                KV_CODEC_ROUTING_PATH=str(run.bundle / "routing.json"),
                KV_CODEC_TREE_PATH=str(run.bundle / "tree_params.pt"),
                KV_CODEC_V_PARAMS_PATH=str(run.bundle / "v_params.pt"),
                KV_CODEC_TEMPORAL_PROFILE_PATH=str(
                    run.bundle / "profiles" / (label(budget) + ".json")
                ),
            )
        if capture:
            env.update(
                KV_CAPTURE_DIR=str(capture),
                KV_CAPTURE_PROMPT_INDEX=str(prompt_id),
                KV_CAPTURE_SAMPLE_TOKENS=str(m["capture_tokens"]),
                KV_CAPTURE_SEED=str(c["calibration"]["seed"]),
            )
        argv = [
            "bash",
            REPO / "scripts/HY-WorldPlay/run_hunyuan8b_bf16.sh",
            "--metrics_path",
            destination / "runtime.json",
        ]
    env.update(extra_env or {})
    video, elapsed = process(argv, env, destination)
    if capture and c["model"] == "wan2_2_5b":
        diag = load_module("scripts/rvq/diag/run_rvq_diag.py")
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
        image_hash = digest(image)
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
        generation_seconds=numeric(
            "gen_time_s", "e2e_seconds", "mp4_complete_elapsed_seconds"
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
        helper = load_module("scripts/HY-WorldPlay/run_moviegen_i2v_job.py")
        summary = helper.summarize_avg2_accounting(
            raw.get("kv_codec_encode_events", []), None
        )
        if summary:
            terminal = summary["terminal_snapshot"]
            result.update(
                k_payload_mib=(
                    terminal["physical_local_k_bytes"]
                    + terminal["physical_shared_k_bytes"]
                )
                / 1024**2,
                v_payload_mib=(
                    terminal["physical_local_v_bytes"]
                    + terminal["physical_shared_v_bytes"]
                )
                / 1024**2,
                bf16_kv_payload_mib=terminal["bf16_kv_bytes"] / 1024**2,
                physical_residual_bpp=terminal["physical_residual_bpp"],
                nominal_residual_avg_bits=terminal["nominal_residual_avg_bits"],
            )
            result["kv_payload_mib"] = terminal["physical_payload_bytes"] / 1024**2
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


def infer(run, budget=None, reference_only=False):
    values = prompts(run.config, "eval")
    budgets = (
        []
        if reference_only
        else ([budget] if budget is not None else run.config["budgets"])
    )
    for selected in [None, *budgets]:
        method = "bf16" if selected is None else label(selected)
        for i, prompt in enumerate(values):

            def inference_one(i=i, prompt=prompt, selected=selected, method=method):
                directory = attempt(run.root / "videos" / method / f"prompt_{i:04d}")
                record = generate(run, i, prompt, "eval", directory, budget=selected)
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
