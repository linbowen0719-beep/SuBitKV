#!/usr/bin/env python3
"""Build a create-only manifest for K-layer-selective frame diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
TENSOR_MANIFEST = (
    REPO_ROOT
    / "configs"
    / "analysis"
    / "wan_closed_loop_frame_diagnostics_p049_20260723.json"
)
GATE_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "k_layer_selective_structured_video_dev_gate_20260724"
)
EVALUATION_DIR = GATE_ROOT / "evaluation_v1_20260724"
RUNTIME_ROOT = (
    DATA_ROOT / "k_layer_selective_structured_video_dev_v1_20260724"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "configs"
    / "analysis"
    / "wan_k_layer_selective_closed_loop_frame_diagnostics_p049_20260724.json"
)
CANDIDATES = {
    "A_mid10": {
        "label": "A: Structured K layers 8-15; QV2 K elsewhere; QV2 V",
        "layers": list(range(8, 16)),
    },
    "B_mid20": {
        "label": "B: Structured K layers 16-23; QV2 K elsewhere; QV2 V",
        "layers": list(range(16, 24)),
    },
    "C_mid10_mid20": {
        "label": "C: Structured K layers 8-23; QV2 K elsewhere; QV2 V",
        "layers": list(range(8, 24)),
    },
}
PROMPTS = (0, 4, 9)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def checked(path: Path, expected: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"{path}: hash mismatch")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def similarity_row(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_json(path)
    if payload.get("schema_version") != "frame_aligned_video_similarity.v2":
        raise ValueError(f"{path}: wrong Similarity schema")
    protocol = payload.get("metric_protocol", {})
    if (
        protocol.get("protocol_id")
        != "frame_aligned_psnr_ssim_lpips_normalized_v2"
        or protocol.get("frame_0_included") is not False
        or protocol.get("lpips", {}).get("normalize_input") is not True
    ):
        raise ValueError(f"{path}: wrong normalized-v2 protocol")
    rows = payload.get("videos")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"{path}: expected one video row")
    return rows[0], checked(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"create-only manifest exists: {output}")

    prior_manifest = load_json(TENSOR_MANIFEST)
    gate_summary_path = EVALUATION_DIR / "summary.json"
    gate_evidence_path = EVALUATION_DIR / "evidence_manifest.json"
    gate_run_info_path = EVALUATION_DIR / "run_info.json"
    gate_summary = load_json(gate_summary_path)
    gate_evidence = load_json(gate_evidence_path)
    gate_run_info = load_json(gate_run_info_path)
    if (
        gate_summary.get("decision") != "STOP_NO_HELDOUT_NO_VBENCH"
        or gate_summary.get("winner") is not None
    ):
        raise ValueError("formal development gate is not the expected negative result")
    for name, path in (
        ("summary.json", gate_summary_path),
        ("evidence_manifest.json", gate_evidence_path),
    ):
        if gate_run_info.get("artifacts", {}).get(name) != sha256_file(path):
            raise ValueError(f"{path}: gate run_info artifact hash mismatch")

    runtime_evidence = {
        (str(item["candidate_id"]), int(item["prompt_id"])): item
        for item in gate_evidence["runtime"]
    }
    similarity_evidence = {
        (str(item["candidate_id"]), int(item["prompt_id"])): item
        for item in gate_evidence["candidate_similarity"]
    }
    expected_keys = {
        (candidate, prompt)
        for candidate in CANDIDATES
        for prompt in PROMPTS
    }
    if set(runtime_evidence) != expected_keys or set(similarity_evidence) != expected_keys:
        raise ValueError("formal gate evidence does not contain the complete 3x3 grid")

    methods = [
        {
            "method_id": candidate,
            "label": payload["label"],
            "role": "candidate",
            "structured_K_layers": payload["layers"],
        }
        for candidate, payload in CANDIDATES.items()
    ]
    prior_methods = {
        str(item["method_id"]): item for item in prior_manifest["methods"]
    }
    for method_id in ("rvq_quantvideo_int2", "uniform_structured_b2"):
        methods.append(
            {
                "method_id": method_id,
                "label": prior_methods[method_id]["label"],
                "role": "comparator",
            }
        )

    prior_prompts = {
        int(item["prompt_id"]): item for item in prior_manifest["prompts"]
    }
    prompts = []
    archived_checks = []
    for prompt_id in PROMPTS:
        prior = prior_prompts[prompt_id]
        prompt_record = {
            "prompt_id": prompt_id,
            "prompt": prior["prompt"],
            "reference": prior["reference"],
            "tests": {},
        }
        for candidate in CANDIDATES:
            evidence = runtime_evidence[(candidate, prompt_id)]
            run_info_path = Path(evidence["run_info_path"]).resolve()
            checked(run_info_path, str(evidence["run_info_sha256"]))
            run_info = load_json(run_info_path)
            if (
                run_info.get("candidate_id") != candidate
                or run_info.get("structured_K_layers")
                != CANDIDATES[candidate]["layers"]
            ):
                raise ValueError(f"{run_info_path}: candidate provenance mismatch")
            video = run_info.get("video_path", {})
            video_path = Path(str(video.get("path", ""))).resolve()
            video_record = checked(video_path, str(video.get("sha256")))
            if (
                video_record["path"] != str(Path(evidence["video_path"]).resolve())
                or video_record["sha256"] != evidence["video_sha256"]
            ):
                raise ValueError(f"{video_path}: runtime/evaluation evidence mismatch")
            similarity_record = similarity_evidence[(candidate, prompt_id)]
            similarity_path = Path(similarity_record["json_path"]).resolve()
            row, similarity_artifact = similarity_row(similarity_path)
            if similarity_artifact["sha256"] != similarity_record["json_sha256"]:
                raise ValueError(f"{similarity_path}: gate evidence hash mismatch")
            if (
                int(row["prompt_id"]) != prompt_id
                or Path(row["test_video"]).resolve() != video_path
                or row.get("test_video_sha256") != video_record["sha256"]
            ):
                raise ValueError(f"{similarity_path}: video identity mismatch")
            prompt_record["tests"][candidate] = {
                **video_record,
                "runtime_run_info": checked(
                    run_info_path, str(evidence["run_info_sha256"])
                ),
                "similarity_source": similarity_artifact,
            }
            archived_checks.append(
                {
                    "prompt_id": prompt_id,
                    "method_id": candidate,
                    "mse_mean": row["mse"],
                    "psnr_db_mean": row["psnr"],
                    "ssim_mean": row["ssim"],
                    "source_path": str(similarity_path),
                    "source_sha256": similarity_artifact["sha256"],
                }
            )
        for method_id in ("rvq_quantvideo_int2", "uniform_structured_b2"):
            prompt_record["tests"][method_id] = prior["tests"][method_id]
            check = next(
                item
                for item in prior_manifest["archived_metric_checks"]
                if int(item["prompt_id"]) == prompt_id
                and item["method_id"] == method_id
            )
            archived_checks.append(check)
        prompts.append(prompt_record)

    manifest = {
        "schema_version": (
            "k_layer_selective_closed_loop_frame_diagnostics.input_manifest.v1"
        ),
        "description": (
            "Read-only frame diagnostics for frozen A/B/C K-layer-selective "
            "development candidates against RVQ+QV2 and uniform Structured B2."
        ),
        "protocol": prior_manifest["protocol"],
        "codebook_path": prior_manifest["codebook_path"],
        "codebook_hash": prior_manifest["codebook_hash"],
        "methods": methods,
        "prompts": prompts,
        "archived_metric_checks": archived_checks,
        "formal_gate": {
            "decision": gate_summary["decision"],
            "winner": gate_summary["winner"],
            "summary": checked(gate_summary_path),
            "evidence_manifest": checked(gate_evidence_path),
            "run_info": checked(gate_run_info_path),
        },
        "source_protocol_manifest": checked(TENSOR_MANIFEST),
        "scope_guards": {
            "development_prompts_only": list(PROMPTS),
            "heldout_prompts_read": False,
            "vbench_run": False,
            "new_video_generation": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(checked(output), indent=2))


if __name__ == "__main__":
    main()
