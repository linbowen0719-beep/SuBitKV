#!/usr/bin/env python3
"""Produce a wording-only clarified P1 final report without changing evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    base = p0.resolve_path(cfg["base_final_dir"])
    root = p0.resolve_path(cfg["output_dir"])
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite clarified final: {root}")
    root.mkdir(parents=True)
    for directory in ("tables", "plots"):
        shutil.copytree(base / directory, root / directory)
    summary = load_json(base / "summary.json")
    summary["schema_version"] = "wan5b_p1_final_research_summary.v4"
    summary["completed"] = p0.utc_now()
    summary["config_hash"] = cfg_hash
    summary["clarification"] = (
        "Wording-only correction: historical v1 limits are explicitly time-scoped; "
        "P1-L2 is supported and the complete map is eligible for decoded-video evaluation."
    )
    p0.atomic_json(root / "summary.json", summary)
    memory = summary["confirmed_full_map_memory"]
    holdout = summary["primary_holdout"]
    confirmation = summary["override_confirmation"]
    primary = confirmation["primary"]
    uniform_layers = confirmation["full_map"]["uniform_l5_c64"]
    decision = [
        "# P1 research decision — clarified final", "",
        "## Recommended research candidate", "",
        f"Use uniform L5C64 at layers `{uniform_layers}` and compact30 `[64,32,16,8,8,8,8,8]` at the other 25 layers. Use actual INT4 centroid payloads with per-centroid FP16 scales and pack the 30 logical index bits in one int32 container.", "",
        "## Evidence", "",
        f"- Four-layer prompt-level holdout: +{100*holdout['mean_prompt_relative_improvement']:.3f}% joint exact-attention improvement vs L5C64, 95% CI [{100*holdout['bootstrap_95_ci_low']:.3f}%, {100*holdout['bootstrap_95_ci_high']:.3f}%], 4/4 wins.", "",
        f"- Independent five-override confirmation on four novel prompts: +{100*primary['mean_relative_improvement']:.3f}% affected-layer joint improvement vs compact30, 95% CI [{100*primary['bootstrap_95_ci_low']:.3f}%, {100*primary['bootstrap_95_ci_high']:.3f}%], 4/4 wins, 5/5 layers favorable.", "",
        f"- Historical index+codebook estimate: {memory['map_int32_index_plus_codebook_mib']:.3f} MiB; {100*memory['reduction_vs_uniform_current_uint8']:.3f}% below the current L5C64 uint8-stage implementation, or {100*memory['reduction_vs_uniform_if_both_int32']:.3f}% below L5C64 if both methods use int32 packing.", "",
        "## Decision boundary", "",
        "P1-L2 is supported, so this map is eligible for integrated runtime and decoded-video evaluation. It is not yet a production recommendation: no decoded-video quality or end-to-end latency result exists, and the eager int32 codec is not integrated into the cache/attention path.",
    ]
    write_text(root / "research_decision.md", "\n".join(decision) + "\n")
    report = (base / "report.md").read_text(encoding="utf-8")
    report = report.replace(
        "- Exact replay covers prompts 0-7, layers 0/10/20/29, and one late-cache call per layer.",
        "- The original prompt-level confirmation covers prompts 0-7 at layers 0/10/20/29; later preregistered addenda cover all 30 layers on prompts 8-9 and independently confirm the five differing layers on four novel prompts.",
    )
    report = report.replace(
        "- The hybrid mapping covers only those four layers; no complete 30-layer runtime policy is claimed.",
        "- At the v1 stage only four layers were mapped. The v3 addendum provides a complete 30-layer research candidate and independent sign confirmation, but no decoded-video or integrated-runtime claim.",
    )
    report += (
        "\n## Clarified final status (v4)\n\n"
        "P1-L2 is supported. The complete map is eligible for decoded-video and integrated-runtime evaluation; it is not a production claim. "
        f"Memory reduction is {100*memory['reduction_vs_uniform_current_uint8']:.3f}% versus the current uint8-stage L5C64 implementation and "
        f"{100*memory['reduction_vs_uniform_if_both_int32']:.3f}% when both schedules use int32 packing.\n"
    )
    write_text(root / "report.md", report)
    base_info = load_json(base / "run_info.json")
    base_summary_path = base / "summary.json"
    base_run_path = base / "run_info.json"
    info = dict(base_info)
    info.update(
        {
            "git_commit": p0.git_commit(), "config_path": str(cfg_path), "config_hash": cfg_hash,
            "timestamp": p0.utc_now(), "model_id": cfg["model_id"], "seed": int(cfg["seed"]),
            "clarification_source": {
                "summary_path": str(base_summary_path.resolve()),
                "summary_sha256": p0.sha256_file(base_summary_path),
                "run_info_path": str(base_run_path.resolve()),
                "run_info_sha256": p0.sha256_file(base_run_path),
            },
        }
    )
    p0.atomic_json(root / "run_info.json", info)
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "run_info.json"}
    )
    p0.atomic_json(
        root / "artifact_manifest.json",
        {
            "schema_version": "wan5b_p1_final_artifact_manifest.v4",
            "files": [
                {"path": str(path.relative_to(root)), "sha256": p0.sha256_file(path), "bytes": path.stat().st_size}
                for path in files
            ],
        },
    )
    print(json.dumps({"output": str(root), "artifacts": len(files)}, indent=2))


if __name__ == "__main__":
    main()
