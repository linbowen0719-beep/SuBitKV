"""Configure HY-WorldPlay's native image-to-video inference process."""

from __future__ import annotations

import sys

from subitkv.pipeline.common import REPO, absolute, write_json


def build_command(
    run,
    prompt_id,
    prompt,
    split,
    destination,
    prompt_file,
    env,
    *,
    budget=None,
    capture=None,
):
    from subitkv.pipeline.adapters import label

    c, m = run.config, run.config["model_config"]
    image = absolute(c["data"]["conditioning_images"]) / split / f"{prompt_id:04d}.png"
    if not image.is_file():
        raise FileNotFoundError(f"HY-WorldPlay I2V input required: {image}")
    options = dict(
        prompt=prompt,
        negative_prompt=m.get("negative_prompt", ""),
        image_path=image,
        output_path=destination,
        model_path=run.models / m["base"],
        action_ckpt=run.models / m["checkpoint"],
        video_length=m["frames"],
        seed=m["seed"],
        num_inference_steps=m["steps"],
        width=m["width"],
        height=m["height"],
        pose=m["pose"],
        resolution="480p",
        aspect_ratio="16:9",
        model_type="ar",
        dtype="bf16",
        rewrite="false",
        sr="false",
        few_step="true",
        offloading="true",
        kv_cache_offload="true",
        full_history_attention="true",
        append_only_vision_cache="true",
        kv_cache_gpu_resident_compressed="false",
        group_offloading="false",
        use_sageattn="false",
        use_fp8_gemm="false",
        use_vae_parallel="false",
        enable_torch_compile="false",
        kv_cache_codec="none",
        kv_codec_frame_spatial_tokens=m["spatial_tokens"],
        kv_capture_history_calls=c["calibration"]["schedule"]["horizon"],
        terminal_cache_audit="true" if budget is not None else "false",
        metrics_path=destination / "runtime.json",
    )
    if budget is not None:
        settings = destination / "runtime_settings.json"
        write_json(settings, c)
        env["SUBITKV_RUNTIME_SETTINGS"] = str(settings)
        options.update(
            kv_cache_codec="rvq_ours_" + label(budget).replace("avg0p2", "avg0p2_alt"),
            kv_codec_k_codebook_dir=run.bundle / "nonorm_k_l16c64_codebooks",
            kv_codec_v_codebook_dir=run.bundle / "nonorm_v_l16c64_codebooks",
            kv_codec_routing_path=run.bundle / "routing.json",
            kv_codec_tree_path=run.bundle / "tree_params.pt",
            kv_codec_v_params_path=run.bundle / "v_params.pt",
            kv_codec_temporal_profile_path=run.bundle
            / "profiles"
            / (label(budget) + ".json"),
        )
    if capture:
        options.update(
            kv_capture_dir=capture,
            kv_capture_prompt_index=prompt_id,
            kv_capture_sample_tokens=m["capture_tokens"],
            kv_capture_seed=c["calibration"]["seed"],
        )
    env["PYTHONPATH"] = f"{REPO}:{REPO / 'third_party/HY-WorldPlay'}"
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--standalone",
        REPO / "third_party/HY-WorldPlay/hyvideo/generate.py",
    ]
    for name, value in options.items():
        argv.extend(["--" + name, str(value)])
    return argv, env
