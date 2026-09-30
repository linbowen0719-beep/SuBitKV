"""Model-specific launch configuration for the public workflow."""

from __future__ import annotations

import sys

import yaml

from subitkv.pipeline.common import REPO, write_json


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
    from subitkv.pipeline.adapters import quant_config

    c, m = run.config, run.config["model_config"]
    base = yaml.safe_load(
        (REPO / "third_party/Self-Forcing/configs/self_forcing_dmd.yaml").read_text()
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
        REPO / "third_party/Self-Forcing/inference.py",
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
        SUBITKV_CKPT_ROOT=str(run.models),
        SELF_FORCING_METRICS_PATH=str(destination / "runtime.json"),
        PYTHONPATH=f"{REPO / 'third_party/Self-Forcing'}:{REPO}",
    )
    return argv, env
