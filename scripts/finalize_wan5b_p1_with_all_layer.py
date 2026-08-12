#!/usr/bin/env python3
"""Create P1 final v2 by adding the all-layer exploratory gate to audited v1."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    all_summary_root = p0.resolve_path(cfg["all_layer_summary_dir"])
    all_exact_root = p0.resolve_path(cfg["all_layer_exact_dir"])
    root = p0.resolve_path(cfg["output_dir"])
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite final v2: {root}")
    required = [
        base / "summary.json", base / "report.md", base / "research_decision.md",
        all_summary_root / "summary.json", all_summary_root / "report.md",
        all_summary_root / "plots" / "all_layer_exact_map.png",
        all_summary_root / "plots" / "all_layer_exact_map.pdf",
        all_exact_root / "tables" / "functional_metrics.csv",
        all_exact_root / "exact_replay" / "codebook_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Final v2 inputs incomplete: {missing}")
    (root / "tables").mkdir(parents=True)
    (root / "plots").mkdir()
    for path in sorted((base / "tables").glob("*")):
        if path.is_file():
            shutil.copy2(path, root / "tables" / path.name)
    for path in sorted((base / "plots").glob("*")):
        if path.is_file():
            shutil.copy2(path, root / "plots" / path.name)
    shutil.copy2(all_summary_root / "plots" / "all_layer_exact_map.png", root / "plots" / "06_all_layer_exact_map.png")
    shutil.copy2(all_summary_root / "plots" / "all_layer_exact_map.pdf", root / "plots" / "06_all_layer_exact_map.pdf")
    for name in ("all_layer_prompt_effects.csv", "policy_prompt_comparisons.csv", "policy_summary.csv"):
        shutil.copy2(all_summary_root / "tables" / name, root / "tables" / name)

    all_summary = load_json(all_summary_root / "summary.json")
    base_summary = load_json(base / "summary.json")
    advance = bool(all_summary["P1-L1_advance_to_video_eval"])
    hypothesis_path = root / "tables" / "hypothesis_outcomes.csv"
    with hypothesis_path.open(newline="", encoding="utf-8") as handle:
        hypotheses = list(csv.DictReader(handle))
    hypotheses.append(
        {
            "id": "P1-L1", "type": "preregistered_exploratory_N2",
            "status": "advance_gate_passed" if advance else "advance_gate_failed",
            "result": (
                f"full-layer frozen hybrid vs uniform {100*all_summary['primary_vs_uniform']['mean_relative_improvement']:.3f}%; "
                f"vs all-compact {100*all_summary['primary_vs_compact']['mean_relative_improvement']:.3f}%; "
                f"sign accuracy p8/p9 {100*all_summary['prediction_accuracy_by_prompt']['8']:.1f}%/"
                f"{100*all_summary['prediction_accuracy_by_prompt']['9']:.1f}%"
            ),
            "decision": "advance frozen 30-layer map to video evaluation" if advance else "do not advance frozen boundary map",
        }
    )
    p0.write_csv(hypothesis_path, hypotheses)
    failure_path = root / "tables" / "failure_log.csv"
    with failure_path.open(newline="", encoding="utf-8") as handle:
        failures = list(csv.DictReader(handle))
    failures.append(
        {
            "event": "final_v1_prompt_list", "status": "provenance_corrected",
            "cause": "v1 merged calibration prompts and holdout 4-7 but omitted exact-development prompt texts 0-3 from run_info.prompt_list",
            "resolution": "v2 enumerates all 20 train and all 10 test prompts; v1 source hashes remain valid",
        }
    )
    p0.write_csv(failure_path, failures)
    key_path = root / "tables" / "key_results.csv"
    with key_path.open(newline="", encoding="utf-8") as handle:
        keys = list(csv.DictReader(handle))
    for result, record in (
        ("full_layer_hybrid_N2_vs_L5", all_summary["primary_vs_uniform"]),
        ("full_layer_hybrid_N2_vs_compact30", all_summary["primary_vs_compact"]),
    ):
        keys.append(
            {
                "result": result,
                "mean_pct": 100 * float(record["mean_relative_improvement"]),
                "ci_low_pct": "not_applicable_N2",
                "ci_high_pct": "not_applicable_N2",
                "prompts": 2,
            }
        )
    p0.write_csv(key_path, keys)

    summary = dict(base_summary)
    summary["schema_version"] = "wan5b_p1_final_research_summary.v2"
    summary["completed"] = p0.utc_now()
    summary["config_hash"] = cfg_hash
    summary["all_layer_exploration"] = all_summary
    summary["hypothesis_outcomes"] = hypotheses
    summary["failures"] = failures
    summary["recommended_research_policy"]["confirmatory_four_layer_policy"] = dict(
        summary["recommended_research_policy"].get("captured_layers", {})
    )
    summary["recommended_research_policy"]["exploratory_full_layer_policy"] = all_summary["frozen_policy"]
    summary["recommended_research_policy"]["full_layer_status"] = (
        "eligible_for_end_to_end_video_evaluation_exploratory_N2"
        if advance else "failed_exploratory_advance_gate"
    )
    summary["hard_limits"] = list(summary["hard_limits"]) + [
        "The full 30-layer boundary map was evaluated on only two prompts; it is an engineering advance gate, not a prompt-generalization result.",
        "Lean all-layer captures validate 64 deterministic post-transform tokens per capture rather than storing the redundant full post-position tensors.",
    ]
    p0.atomic_json(root / "summary.json", summary)

    primary = base_summary["primary_holdout"]
    full_u = all_summary["primary_vs_uniform"]
    full_c = all_summary["primary_vs_compact"]
    policy_text = "L5C64 at layers 0-4 and 25-29; compact30 at layers 5-24"
    decision = [
        "# P1 research decision v2", "",
        "## Confirmatory result", "",
        f"The four-layer policy remains the strongest prompt-level result: +{100*primary['mean_prompt_relative_improvement']:.3f}% joint exact-attention improvement vs L5C64 on untouched prompts 4-7 (95% CI [{100*primary['bootstrap_95_ci_low']:.3f}%, {100*primary['bootstrap_95_ci_high']:.3f}%], 4/4 wins).", "",
        "## Full-layer exploratory gate", "",
        f"Frozen map: {policy_text}.", "",
        f"On the last untouched prompts 8-9 it changes joint error by {100*full_u['mean_relative_improvement']:+.3f}% vs all-uniform and {100*full_c['mean_relative_improvement']:+.3f}% vs all-compact. Layer-sign accuracy is {100*all_summary['prediction_accuracy_by_prompt']['8']:.1f}%/{100*all_summary['prediction_accuracy_by_prompt']['9']:.1f}%.", "",
        f"Advance-to-video-evaluation gate: **{'passed' if advance else 'failed'}**.", "",
        "## Boundary", "",
        "Even if the gate passes, N=2 is insufficient for a prompt-level full-layer claim. The next authorized step is end-to-end decoded-video quality and integrated runtime latency, followed by a larger prompt confirmation—not production deployment.",
    ]
    write_text(root / "research_decision.md", "\n".join(decision) + "\n")
    base_report = (base / "report.md").read_text(encoding="utf-8").rstrip()
    addendum = [
        "", "## All-layer addendum (v2)", "",
        f"After the v1 report was frozen, prompts 8-9 were used for a preregistered exploratory all-30-layer map. The policy `{policy_text}` was frozen before capture.", "",
        f"Its joint exact-attention effect is {100*full_u['mean_relative_improvement']:+.3f}% vs all-uniform and {100*full_c['mean_relative_improvement']:+.3f}% vs all-compact. The two prompt-specific ranges are [{100*full_u['min_prompt_improvement']:.3f}%, {100*full_u['max_prompt_improvement']:.3f}%] and [{100*full_c['min_prompt_improvement']:.3f}%, {100*full_c['max_prompt_improvement']:.3f}%], respectively.", "",
        f"The preregistered N=2 engineering gate **{'passed' if advance else 'failed'}**. This is not a new confidence-interval claim. See `plots/06_all_layer_exact_map.*` and the `all_layer_*` tables.", "",
        "All 60 lean references used complete BF16 pre-position K/V, exact stored transform metadata, and 64 deterministic post-position validation tokens; validation and current-segment invariance must pass for every capture.",
    ]
    write_text(root / "report.md", base_report + "\n" + "\n".join(addendum) + "\n")

    train_prompts = prompt_lines(p0.resolve_path(cfg["prompt_train_file"]))
    test_prompts = prompt_lines(p0.resolve_path(cfg["prompt_test_file"]))
    source_paths = {
        "base_final_summary": base / "summary.json",
        "base_final_run_info": base / "run_info.json",
        "all_layer_summary": all_summary_root / "summary.json",
        "all_layer_metrics": all_exact_root / "tables" / "functional_metrics.csv",
        "all_layer_codebook_manifest": all_exact_root / "exact_replay" / "codebook_manifest.json",
    }
    source_records = {
        name: {"path": str(path.resolve()), "sha256": p0.sha256_file(path), "bytes": path.stat().st_size}
        for name, path in source_paths.items()
    }
    base_info = load_json(base / "run_info.json")
    registry = p0.resolve_path(cfg["hypothesis_registry"])
    codebook_paths = dict(base_info["codebook_path"])
    codebook_hashes = dict(base_info["codebook_hash"])
    codebook_paths["all_layer_exact"] = str(source_paths["all_layer_codebook_manifest"].resolve())
    codebook_hashes["all_layer_exact"] = source_records["all_layer_codebook_manifest"]["sha256"]
    p0.atomic_json(
        root / "run_info.json",
        {
            "git_commit": p0.git_commit(), "config_path": str(cfg_path), "config_hash": cfg_hash,
            "codebook_path": codebook_paths, "codebook_hash": codebook_hashes,
            "hostname": p0.socket.gethostname(), "gpu_type": p0.gpu_name(), "seed": int(cfg["seed"]),
            "prompt_list": train_prompts + test_prompts, "calibration_prompt_list": train_prompts,
            "functional_prompt_list": test_prompts, "timestamp": p0.utc_now(), "model_id": cfg["model_id"],
            "source_records": source_records, "hypothesis_registry": str(registry.resolve()),
            "hypothesis_registry_hash": p0.sha256_file(registry),
            "provenance_correction": "enumerates all train20 and test10 prompts",
        },
    )
    artifact_files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "run_info.json"}
    )
    p0.atomic_json(
        root / "artifact_manifest.json",
        {
            "schema_version": "wan5b_p1_final_artifact_manifest.v2",
            "files": [
                {"path": str(path.relative_to(root)), "sha256": p0.sha256_file(path), "bytes": path.stat().st_size}
                for path in artifact_files
            ],
        },
    )
    print(json.dumps({"output": str(root), "advance": advance, "artifacts": len(artifact_files)}, indent=2))


if __name__ == "__main__":
    main()
