#!/usr/bin/env python3
"""Run the complete matched-budget Value study on an explicit GPU pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
WAN_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
SELF_PYTHON = Path("/home/bowen/QuantVid/quantvid/bin/python")
WAN_CAL = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
SELF_CAL = Path("/data/bowen/self_forcing_96latent_rvq_l8c64")
WAN_REUSE = REPO / "results/long_video/no_scale_variants_11chunk_10prompt_20260707"
WAN_SENTINEL = REPO / "results/long_video/value_channel_reuse_sentinel_20260716"
WAN_NEW = REPO / "results/long_video/value_token_channel_matched_20260716"
SELF_REFERENCE = Path("/data/bowen/self_forcing_96latent_moviegen")
SELF_NEW = Path("/data/bowen/self_forcing_96latent_value_matched_20260716")
SUMMARY = REPO / "results/value_token_channel_matched_20260716"
CONFIG = REPO / "experiments/long_video/value_token_channel_matched_11chunk_10prompts.yaml"

CHANNEL_SETTINGS = (
    "nonorm_k_token_v_channel_l8c64",
    "first_recent_nonorm_k_token_v_channel_l8c64",
)
TOKEN_SETTINGS = (
    "nonorm_k_token_l8_v_token_l32c64",
    "first_recent_nonorm_k_token_l8_v_token_l32c64",
)


def call(stage: str, command: list[str], *, check: bool = True) -> int:
    print(f"\n[stage] {stage}\n[command] {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=REPO)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def parallel(stage: str, jobs: list[tuple[str, list[str]]]) -> None:
    running = []
    print(f"\n[stage] {stage}", flush=True)
    for name, command in jobs:
        print(f"[launch] {name}: {' '.join(command)}", flush=True)
        running.append((name, command, subprocess.Popen(command, cwd=REPO)))
    failures = []
    for name, command, process in running:
        code = process.wait()
        print(f"[finish] {name}: returncode={code}", flush=True)
        if code:
            failures.append((name, command, code))
    if failures:
        raise RuntimeError(f"Parallel stage failed: {failures}")


def wan_job(setting: str, gpu: str, run_id: str, *, prompts: int, vbench: bool) -> list[str]:
    command = [
        "/usr/bin/env", f"MASTER_PORT={29600 + int(gpu)}",
        str(WAN_PYTHON), str(REPO / "scripts/benchmark_long_video.py"),
        "--config", str(CONFIG), "--output-root", str(REPO / "results/long_video"),
        "--run-id", run_id, "--settings", setting, "--gpus", gpu,
        "--execute", "--allow-full-run", "--resume", "--no-aggregate",
    ]
    if prompts == 1:
        command.extend(["--prompt-start-override", "0", "--num-prompts-override", "1"])
    if vbench:
        command.append("--run_vbench")
    else:
        command.append("--skip_vbench")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="5,7")
    args = parser.parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) != 2:
        raise ValueError("This pipeline requires exactly two GPU ids")

    call("wan_codebooks", [
        str(WAN_PYTHON), str(REPO / "scripts/prepare_value_token_channel_codebooks.py"),
        "--model", "wan5b", "--root", str(WAN_CAL),
        "--prompt-source", str(WAN_CAL / "prompts_train.txt"),
        "--python", str(WAN_PYTHON), "--gpus", ",".join(gpus), "--stage", "all",
    ])
    call("wan_codec_smoke", [
        str(WAN_PYTHON), str(REPO / "scripts/smoke_value_token_channel_codecs.py"),
        "--model", "wan5b", "--calibration-root", str(WAN_CAL),
        "--token-v-codebooks", str(WAN_CAL / "value_token_nonorm_v_l32c64_codebooks"),
        "--channel-v-codebooks", str(WAN_CAL / "grouped_direct_nonorm_v_l8c64_codebooks"),
        "--output-dir", str(REPO / "results/value_token_channel_codec_smoke/wan5b"),
    ])
    parallel("wan_channel_sentinels", [
        (setting, wan_job(setting, gpu, WAN_SENTINEL.name, prompts=1, vbench=False))
        for setting, gpu in zip(CHANNEL_SETTINGS, gpus)
    ])

    audit_path = WAN_SENTINEL / "reuse_audit.json"
    if not audit_path.exists():
        call("wan_reuse_audit", [
            str(WAN_PYTHON), str(REPO / "scripts/audit_wan_value_channel_reuse.py"),
            "--sentinel-root", str(WAN_SENTINEL), "--reuse-root", str(WAN_REUSE),
            "--output", str(audit_path),
        ], check=False)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    failed_channel = [name for name in CHANNEL_SETTINGS if not audit["settings"][name]["reusable"]]
    if failed_channel:
        parallel("wan_failed_channel_fallback", [
            (setting, wan_job(setting, gpus[index % 2], WAN_NEW.name, prompts=10, vbench=True))
            for index, setting in enumerate(failed_channel)
        ])

    parallel("wan_token_formal", [
        (setting, wan_job(setting, gpu, WAN_NEW.name, prompts=10, vbench=True))
        for setting, gpu in zip(TOKEN_SETTINGS, gpus)
    ])

    call("self_codebooks", [
        str(SELF_PYTHON), str(REPO / "scripts/prepare_value_token_channel_codebooks.py"),
        "--model", "self_forcing_1p3b", "--root", str(SELF_CAL),
        "--prompt-source", str(WAN_CAL / "prompts_train.txt"),
        "--python", str(SELF_PYTHON), "--gpus", ",".join(gpus), "--stage", "all",
    ])
    call("self_codec_smoke", [
        str(SELF_PYTHON), str(REPO / "scripts/smoke_value_token_channel_codecs.py"),
        "--model", "self_forcing_1p3b", "--calibration-root", str(SELF_CAL),
        "--token-v-codebooks", str(SELF_CAL / "value_token_nonorm_v_l32c64_codebooks"),
        "--channel-v-codebooks", str(SELF_CAL / "grouped_direct_nonorm_v_l8c64_codebooks"),
        "--output-dir", str(REPO / "results/value_token_channel_codec_smoke/self_forcing_1p3b"),
    ])
    call("self_formal", [
        str(SELF_PYTHON), str(REPO / "scripts/Self-Forcing/watch_96latent_experiment.py"),
        "--value-study", "--output-root", str(SELF_NEW), "--rvq-root", str(SELF_CAL),
        "--value-token-v-codebooks", str(SELF_CAL / "value_token_nonorm_v_l32c64_codebooks"),
        "--value-channel-v-codebooks", str(SELF_CAL / "grouped_direct_nonorm_v_l8c64_codebooks"),
        "--reference-root", str(SELF_REFERENCE), "--python", str(SELF_PYTHON),
        "--gpus", ",".join(gpus),
    ])
    call("paired_summary", [
        str(WAN_PYTHON), str(REPO / "scripts/summarize_value_token_channel_study.py"),
        "--wan-new-root", str(WAN_NEW), "--wan-reuse-root", str(WAN_REUSE),
        "--wan-sentinel-root", str(WAN_SENTINEL), "--wan-reuse-audit", str(audit_path),
        "--self-root", str(SELF_NEW), "--output-dir", str(SUMMARY),
    ])
    print("[complete] matched-budget Value study", flush=True)


if __name__ == "__main__":
    main()
