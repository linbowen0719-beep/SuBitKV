#!/usr/bin/env python3
"""Create final P1 v3 with independent confirmation of the full-layer overrides."""

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


def prompts(path: Path) -> list[str]:
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
    confirmation = p0.resolve_path(cfg["confirmation_summary_dir"])
    exact = p0.resolve_path(cfg["confirmation_exact_dir"])
    root = p0.resolve_path(cfg["output_dir"])
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite final v3: {root}")
    required = [
        base / "summary.json", base / "report.md", base / "research_decision.md",
        confirmation / "summary.json", confirmation / "report.md",
        confirmation / "plots" / "override_confirmation.png",
        confirmation / "plots" / "override_confirmation.pdf",
        exact / "tables" / "functional_metrics.csv",
        exact / "exact_replay" / "codebook_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Final v3 inputs incomplete: {missing}")
    (root / "tables").mkdir(parents=True)
    (root / "plots").mkdir()
    for path in (base / "tables").glob("*"):
        if path.is_file():
            shutil.copy2(path, root / "tables" / path.name)
    for path in (base / "plots").glob("*"):
        if path.is_file():
            shutil.copy2(path, root / "plots" / path.name)
    shutil.copy2(confirmation / "plots" / "override_confirmation.png", root / "plots" / "07_override_confirmation.png")
    shutil.copy2(confirmation / "plots" / "override_confirmation.pdf", root / "plots" / "07_override_confirmation.pdf")
    for name in ("prompt_comparison.csv", "intervention_summary.csv", "layer_confirmation.csv"):
        shutil.copy2(confirmation / "tables" / name, root / "tables" / f"override_{name}")

    conf = load_json(confirmation / "summary.json")
    base_summary = load_json(base / "summary.json")
    supported = bool(conf["P1-L2_supported"])
    primary = conf["primary"]
    hypothesis_file = root / "tables" / "hypothesis_outcomes.csv"
    with hypothesis_file.open(newline="", encoding="utf-8") as handle:
        hypotheses = list(csv.DictReader(handle))
    hypotheses.append(
        {
            "id": "P1-L2", "type": "confirmatory_novel_prompts",
            "status": "supported" if supported else "not_supported",
            "result": (
                f"five uniform overrides vs compact30 on affected layers "
                f"{100*primary['mean_relative_improvement']:+.3f}% "
                f"CI [{100*primary['bootstrap_95_ci_low']:.3f}%, {100*primary['bootstrap_95_ci_high']:.3f}%], "
                f"{100*primary['prompt_win_rate']:.1f}% prompt wins, {conf['favorable_layer_count']}/5 layers favorable"
            ),
            "decision": "advance confirmed 30-layer map to decoded-video evaluation" if supported else "retain all-compact pending a new mapping",
        }
    )
    p0.write_csv(hypothesis_file, hypotheses)
    key_file = root / "tables" / "key_results.csv"
    with key_file.open(newline="", encoding="utf-8") as handle:
        key_rows = list(csv.DictReader(handle))
    key_rows.append(
        {
            "result": "five_override_confirmation_vs_compact30",
            "mean_pct": 100*primary["mean_relative_improvement"],
            "ci_low_pct": 100*primary["bootstrap_95_ci_low"],
            "ci_high_pct": 100*primary["bootstrap_95_ci_high"],
            "prompts": 4,
        }
    )
    p0.write_csv(key_file, key_rows)
    full_map = conf["full_map"]
    uniform_layers = [int(v) for v in full_map["uniform_l5_c64"]]
    compact_layers = [int(v) for v in full_map["compact30"]]
    if sorted(uniform_layers + compact_layers) != list(range(30)):
        raise ValueError("Confirmed full map does not cover all layers")
    # All-layer codebook payload is linear in the number of layer-specific centroids.
    uniform_centroids = 320
    compact_centroids = 152
    map_centroids = len(uniform_layers)*uniform_centroids + len(compact_layers)*compact_centroids
    baseline_centroids = 30*uniform_centroids
    uniform_codebook_mib = 29.00390625
    map_codebook_mib = uniform_codebook_mib * map_centroids / baseline_centroids
    int32_indices_mib = 193.359375
    uniform_current_total_mib = 270.703125
    memory = {
        "scope": "all 30 layers historical K+V; excludes current BF16 and workspace",
        "uniform_override_layers": uniform_layers, "compact_layers": compact_layers,
        "codebook_centroid_reduction_vs_uniform": 1-map_centroids/baseline_centroids,
        "map_int4_codebook_mib": map_codebook_mib,
        "map_int32_index_plus_codebook_mib": int32_indices_mib + map_codebook_mib,
        "reduction_vs_uniform_current_uint8": 1-(int32_indices_mib+map_codebook_mib)/uniform_current_total_mib,
        "reduction_vs_uniform_if_both_int32": 1-(int32_indices_mib+map_codebook_mib)/(int32_indices_mib+uniform_codebook_mib),
    }
    summary = dict(base_summary)
    summary["schema_version"] = "wan5b_p1_final_research_summary.v3"
    summary["completed"] = p0.utc_now()
    summary["config_hash"] = cfg_hash
    summary["override_confirmation"] = conf
    summary["hypothesis_outcomes"] = hypotheses
    summary["confirmed_full_map_memory"] = memory
    summary["recommended_research_policy"]["full_layer_policy"] = {
        "uniform_l5_c64": uniform_layers, "compact30": compact_layers,
    }
    summary["recommended_research_policy"]["full_layer_status"] = (
        "independently_confirmed_attention_direction_N4_eligible_for_decoded_video_evaluation"
        if supported else "not_supported_retain_all_compact"
    )
    summary["hard_limits"] = list(summary["hard_limits"]) + [
        "P1-L2 confirms only the sign contributed by the five differing layers; it does not measure decoded-video quality or integrated runtime.",
    ]
    p0.atomic_json(root / "summary.json", summary)

    holdout = base_summary["primary_holdout"]
    decision = [
        "# P1 research decision v3", "",
        "## Stable prompt-level result", "",
        f"The original four-layer frozen policy remains supported on prompts 4-7: +{100*holdout['mean_prompt_relative_improvement']:.3f}% joint exact-attention improvement vs L5C64 (95% CI [{100*holdout['bootstrap_95_ci_low']:.3f}%, {100*holdout['bootstrap_95_ci_high']:.3f}%], 4/4 wins).", "",
        "## Complete 30-layer candidate", "",
        f"Uniform L5C64 overrides: `{uniform_layers}`; compact30 on the other 25 layers.", "",
        f"Independent P1-L2 verdict on four novel prompts: **{'supported' if supported else 'not supported'}**. Affected-layer joint improvement vs all-compact is {100*primary['mean_relative_improvement']:+.3f}% (95% CI [{100*primary['bootstrap_95_ci_low']:.3f}%, {100*primary['bootstrap_95_ci_high']:.3f}%], {100*primary['prompt_win_rate']:.1f}% prompt wins; {conf['favorable_layer_count']}/5 layers favorable).", "",
        f"Physical historical index+codebook estimate with int32 packing: {memory['map_int32_index_plus_codebook_mib']:.3f} MiB, {100*memory['reduction_vs_uniform_current_uint8']:.3f}% below current uniform L5C64 uint8-stage storage.", "",
        "## Next authorized gate", "",
        "If P1-L2 is supported, integrate the int32 codec and this map into the runtime, then run decoded-video quality plus end-to-end latency on a preregistered prompt set. Exact attention evidence is necessary but not sufficient for production.",
    ]
    write_text(root / "research_decision.md", "\n".join(decision) + "\n")
    base_report = (base / "report.md").read_text(encoding="utf-8").rstrip()
    addendum = [
        "", "## Independent full-map confirmation (v3)", "",
        "The all-layer N=2 study discovered a five-layer uniform override set `{0,1,23,25,29}`. That set was frozen before four new prompts were written and generated.", "",
        f"P1-L2 **{'supports' if supported else 'does not support'}** the overrides: {100*primary['mean_relative_improvement']:+.3f}% affected-layer joint improvement vs compact30, 95% prompt-bootstrap CI [{100*primary['bootstrap_95_ci_low']:.3f}%, {100*primary['bootstrap_95_ci_high']:.3f}%], {100*primary['prompt_win_rate']:.1f}% prompt wins, {conf['favorable_layer_count']}/5 affected layers favorable.", "",
        "The remaining 25 layers are compact30 in both candidate and baseline, so they cancel in the equal-layer-average difference. This validates the direction of the complete map but not video quality or runtime.", "",
        "See `plots/07_override_confirmation.*` and `tables/override_*`.",
    ]
    write_text(root / "report.md", base_report + "\n" + "\n".join(addendum) + "\n")

    train = prompts(p0.resolve_path(cfg["prompt_train_file"]))
    test = prompts(p0.resolve_path(cfg["prompt_test_file"]))
    novel = prompts(p0.resolve_path(cfg["prompt_novel_file"]))
    source_paths = {
        "base_final_v2_summary": base / "summary.json",
        "base_final_v2_run_info": base / "run_info.json",
        "override_summary": confirmation / "summary.json",
        "override_metrics": exact / "tables" / "functional_metrics.csv",
        "override_codebook_manifest": exact / "exact_replay" / "codebook_manifest.json",
    }
    records = {
        name: {"path": str(path.resolve()), "sha256": p0.sha256_file(path), "bytes": path.stat().st_size}
        for name, path in source_paths.items()
    }
    base_info = load_json(base / "run_info.json")
    codebook_path = dict(base_info["codebook_path"])
    codebook_hash = dict(base_info["codebook_hash"])
    codebook_path["override_confirmation"] = str(source_paths["override_codebook_manifest"].resolve())
    codebook_hash["override_confirmation"] = records["override_codebook_manifest"]["sha256"]
    registry = p0.resolve_path(cfg["hypothesis_registry"])
    p0.atomic_json(
        root / "run_info.json",
        {
            "git_commit": p0.git_commit(), "config_path": str(cfg_path), "config_hash": cfg_hash,
            "codebook_path": codebook_path, "codebook_hash": codebook_hash,
            "hostname": p0.socket.gethostname(), "gpu_type": p0.gpu_name(), "seed": int(cfg["seed"]),
            "prompt_list": train+test+novel, "calibration_prompt_list": train,
            "functional_prompt_list": test+novel, "novel_confirmation_prompt_list": novel,
            "timestamp": p0.utc_now(), "model_id": cfg["model_id"], "source_records": records,
            "hypothesis_registry": str(registry.resolve()), "hypothesis_registry_hash": p0.sha256_file(registry),
        },
    )
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "run_info.json"}
    )
    p0.atomic_json(
        root / "artifact_manifest.json",
        {
            "schema_version": "wan5b_p1_final_artifact_manifest.v3",
            "files": [
                {"path": str(path.relative_to(root)), "sha256": p0.sha256_file(path), "bytes": path.stat().st_size}
                for path in files
            ],
        },
    )
    print(json.dumps({"output": str(root), "supported": supported, "artifacts": len(files)}, indent=2))


if __name__ == "__main__":
    main()
