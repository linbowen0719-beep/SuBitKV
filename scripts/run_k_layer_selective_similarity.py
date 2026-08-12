#!/usr/bin/env python3
"""Run one protocol-safe Similarity unit for the K-layer dev screen.

The script has a deliberately strict two-phase gate:

1. all A/B/C x prompt-0/4/9 ``run_info.json`` and generation-side
   ``gate_audit.json`` files must exist, and every audit must say ``PASS``;
2. only after that metadata-only gate succeeds may the script hash/open any
   candidate video or compute Similarity.

Each invocation evaluates exactly one candidate/prompt pair.  Independent
invocations can therefore be distributed over GPUs.  The output remains
compatible with ``analyze_k_layer_selective_video_dev.py --similarity
CANDIDATE=JSON`` while adding immutable BF16/candidate video hashes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import analyze_k_layer_selective_video_dev as dev_gate  # noqa: E402


DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_RUNTIME_ROOT = (
    DATA_ROOT / "k_layer_selective_structured_video_dev_v1_20260724"
)
DEFAULT_DESIGN_DIR = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "k_layer_selective_structured_dev_design_20260724"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "k_layer_selective_structured_video_dev_gate_20260724"
    / "candidate_similarity_normalized_v2"
)

EXPECTED_CANDIDATES = dev_gate.EXPECTED_CANDIDATES
EXPECTED_PROMPTS = dev_gate.EXPECTED_PROMPTS
SEED = 42
GENERATION_AUDIT_SCHEMA = "wan5b_k_layer_selective_video_dev.audit.v1"
GENERATION_RUN_INFO_SCHEMA = (
    "wan5b_k_layer_selective_video_dev.run_info.v1"
)
SIMILARITY_SCHEMA = "frame_aligned_video_similarity.v2"
SIMILARITY_PROTOCOL = "frame_aligned_psnr_ssim_lpips_normalized_v2"
SIMILARITY_RUN_INFO_SCHEMA = (
    "k_layer_selective_video_similarity.run_info.v1"
)
SIMILARITY_CONFIG_SCHEMA = (
    "k_layer_selective_video_similarity.config.v1"
)
EXPECTED_EVALUATOR_SHA256 = (
    "2d0d884b6dfea96b2d5296e4d07036b3ebad3ad3ac346d9ba4517103e1e0046e"
)
EVALUATOR_PATH = REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"
HASH_CHARS = frozenset("0123456789abcdef")

BF16_REFERENCES: dict[int, dict[str, Any]] = {
    0: {
        "path": (
            DATA_ROOT / "videos" / "bf16" / "0-42.mp4"
        ),
        "sha256": (
            "296a43b166289e4603a2b6f1f462e51b48738a50d382566a1e8b12f8742739e5"
        ),
        "bytes": 7_390_463,
    },
    4: {
        "path": (
            DATA_ROOT / "videos" / "bf16" / "4-42.mp4"
        ),
        "sha256": (
            "57bccdd03a5913e07afa7fdc52d49bbf1e5843ca57ca7542ea9e07131516a452"
        ),
        "bytes": 5_880_838,
    },
    9: {
        "path": (
            DATA_ROOT / "videos" / "bf16" / "9-42.mp4"
        ),
        "sha256": (
            "32b4d63b00692d0bc7c0c417c8b70bf5f7f45a6440829d4afb9cc8450c702569"
        ),
        "bytes": 2_135_762,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Compute and publish exactly one Similarity unit.",
    )
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="Run the nine-unit runtime gate without computing quality.",
    )
    parser.add_argument(
        "--candidate", choices=EXPECTED_CANDIDATES, default="A_mid10"
    )
    parser.add_argument(
        "--prompt-id", type=int, choices=EXPECTED_PROMPTS, default=0
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index used by this one Similarity unit.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="CPU is intended only for smoke/debug execution.",
    )
    parser.add_argument(
        "--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT
    )
    parser.add_argument(
        "--runtime",
        action="append",
        default=None,
        metavar="CANDIDATE=ROOT",
        help=(
            "Optional explicit runtime mapping. Repeat exactly nine times; "
            "otherwise paths are derived from --runtime-root."
        ),
    )
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _valid_hash(value: Any, *, field: str, source: Path) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in HASH_CHARS for char in text):
        raise ValueError(f"{source}: invalid {field}: {text!r}")
    return text


def _record_path_hash(
    record: Any, *, field: str, source: Path
) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{source}: {field} record is missing")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    digest = _valid_hash(
        record.get("sha256"), field=f"{field}.sha256", source=source
    )
    return path, digest


def _write_json_create_only(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _runtime_mappings(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.runtime:
        return dev_gate._parse_mapping(list(args.runtime), label="runtime")
    root = args.runtime_root.expanduser().resolve()
    return [
        (
            candidate,
            root / f"k_layer_{candidate}_p{prompt_id}",
        )
        for candidate in EXPECTED_CANDIDATES
        for prompt_id in EXPECTED_PROMPTS
    ]


def _prompt_record(run_info: Mapping[str, Any], *, source: Path) -> dict[str, Any]:
    prompt_list = run_info.get("prompt_list")
    if not isinstance(prompt_list, list) or len(prompt_list) != 1:
        raise ValueError(f"{source}: prompt_list must contain exactly one row")
    record = prompt_list[0]
    if not isinstance(record, Mapping):
        raise ValueError(f"{source}: malformed prompt_list row")
    return {
        "prompt_id": int(record.get("prompt_id", -1)),
        "prompt": str(record.get("prompt", "")),
    }


def _check_nonvideo_output(
    *,
    key: str,
    gate_audit: Mapping[str, Any],
    run_info: Mapping[str, Any],
    gate_path: Path,
    run_info_path: Path,
) -> dict[str, str]:
    gate_outputs = gate_audit.get("outputs")
    if not isinstance(gate_outputs, Mapping):
        raise ValueError(f"{gate_path}: outputs object is missing")
    path, expected_hash = _record_path_hash(
        gate_outputs.get(key), field=f"outputs.{key}", source=gate_path
    )
    if path.suffix.lower() == ".mp4":
        raise AssertionError("metadata gate attempted to open a video")
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{path}: generation audit {key} hash mismatch")

    if key == "config":
        declared_path = Path(
            str(run_info.get("config_path", ""))
        ).expanduser().resolve()
        declared_hash = _valid_hash(
            run_info.get("config_hash"),
            field="config_hash",
            source=run_info_path,
        )
    else:
        declared_path, declared_hash = _record_path_hash(
            run_info.get(key), field=key, source=run_info_path
        )
    if (declared_path, declared_hash) != (path, expected_hash):
        raise ValueError(
            f"{run_info_path}: {key} differs from generation gate audit"
        )
    return {"path": str(path), "sha256": expected_hash}


def metadata_runtime_gate(
    mappings: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    """Validate the complete PASS grid without opening any candidate video."""

    if len(mappings) != 9:
        raise ValueError(f"runtime gate expected 9 entries, got {len(mappings)}")

    # Presence is checked for the full grid before even one audit is trusted.
    required: list[Path] = []
    for _, root in mappings:
        required.extend([root / "run_info.json", root / "gate_audit.json"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "runtime gate incomplete before candidate-video access; missing: "
            + ", ".join(missing)
        )

    # Load only JSON metadata and require all nine PASS statuses first.
    loaded: list[dict[str, Any]] = []
    for mapped_candidate, root in mappings:
        run_info_path = root / "run_info.json"
        gate_path = root / "gate_audit.json"
        run_info = _load_json(run_info_path)
        gate_audit = _load_json(gate_path)
        if run_info.get("schema_version") != GENERATION_RUN_INFO_SCHEMA:
            raise ValueError(f"{run_info_path}: wrong run_info schema")
        if gate_audit.get("schema_version") != GENERATION_AUDIT_SCHEMA:
            raise ValueError(f"{gate_path}: wrong generation audit schema")
        if gate_audit.get("status") != "PASS":
            raise RuntimeError(
                f"{gate_path}: runtime audit is not PASS; candidate videos "
                "remain sealed"
            )
        loaded.append(
            {
                "mapped_candidate": mapped_candidate,
                "root": root.resolve(),
                "run_info_path": run_info_path.resolve(),
                "gate_path": gate_path.resolve(),
                "run_info": run_info,
                "gate_audit": gate_audit,
            }
        )

    rows: list[dict[str, Any]] = []
    for item in loaded:
        candidate = item["mapped_candidate"]
        root = item["root"]
        run_info_path = item["run_info_path"]
        gate_path = item["gate_path"]
        run_info = item["run_info"]
        gate_audit = item["gate_audit"]
        prompt = _prompt_record(run_info, source=run_info_path)
        prompt_id = prompt["prompt_id"]
        if candidate not in EXPECTED_CANDIDATES:
            raise ValueError(f"{run_info_path}: unknown candidate {candidate}")
        if prompt_id not in EXPECTED_PROMPTS:
            raise ValueError(f"{run_info_path}: unexpected prompt {prompt_id}")
        if run_info.get("candidate_id") != candidate:
            raise ValueError(f"{run_info_path}: candidate_id mismatch")
        if (
            gate_audit.get("candidate_id") != candidate
            or int(gate_audit.get("prompt_id", -1)) != prompt_id
        ):
            raise ValueError(f"{gate_path}: candidate/prompt mismatch")
        if int(run_info.get("seed", -1)) != SEED:
            raise ValueError(f"{run_info_path}: seed mismatch")

        run_video_path, run_video_hash = _record_path_hash(
            run_info.get("video_path"),
            field="video_path",
            source=run_info_path,
        )
        gate_outputs = gate_audit.get("outputs")
        if not isinstance(gate_outputs, Mapping):
            raise ValueError(f"{gate_path}: outputs object is missing")
        gate_video_path, gate_video_hash = _record_path_hash(
            gate_outputs.get("video"),
            field="outputs.video",
            source=gate_path,
        )
        if (run_video_path, run_video_hash) != (
            gate_video_path,
            gate_video_hash,
        ):
            raise ValueError(
                f"{gate_path}: video declaration differs from run_info"
            )
        expected_video = (root / f"{prompt_id}-{SEED}.mp4").resolve()
        if run_video_path != expected_video:
            raise ValueError(
                f"{run_info_path}: candidate video path is not canonical"
            )

        nonvideo = {
            key: _check_nonvideo_output(
                key=key,
                gate_audit=gate_audit,
                run_info=run_info,
                gate_path=gate_path,
                run_info_path=run_info_path,
            )
            for key in ("config", "metrics", "generation_log")
        }
        rows.append(
            {
                "candidate_id": candidate,
                "prompt_id": prompt_id,
                "prompt": prompt["prompt"],
                "root": str(root),
                "run_info_path": str(run_info_path),
                "run_info_sha256": sha256_file(run_info_path),
                "gate_audit_path": str(gate_path),
                "gate_audit_sha256": sha256_file(gate_path),
                "gate_status": "PASS",
                "video_path": str(run_video_path),
                # This is only the generation-side declaration.  The bytes
                # remain unopened until all nine rows have passed this phase.
                "declared_video_sha256": run_video_hash,
                "nonvideo_outputs": nonvideo,
            }
        )

    actual = {(row["candidate_id"], row["prompt_id"]) for row in rows}
    expected = {
        (candidate, prompt_id)
        for candidate in EXPECTED_CANDIDATES
        for prompt_id in EXPECTED_PROMPTS
    }
    if actual != expected or len(rows) != len(actual):
        raise ValueError(
            "runtime gate is not the exact A/B/C x prompt-0/4/9 grid"
        )
    return sorted(
        rows,
        key=lambda row: (
            EXPECTED_CANDIDATES.index(row["candidate_id"]),
            row["prompt_id"],
        ),
    )


def full_runtime_audit(
    mappings: list[tuple[str, Path]],
    design: Mapping[str, Any],
    metadata_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replay the formal audit after the metadata-only nine-PASS barrier."""

    audited = dev_gate._audit_all_runtimes(
        mappings,
        candidates=design["candidates"],
        protocol_path=design["protocol_path"],
        amendment_path=design["amendment_path"],
        amendment_sha256=design["amendment_sha256"],
        frozen_source_sha256=design["frozen_source_sha256"],
    )
    metadata = {
        (row["candidate_id"], row["prompt_id"]): row for row in metadata_rows
    }
    for row in audited:
        key = (row["candidate_id"], row["prompt_id"])
        declared = metadata[key]
        if Path(row["video_path"]).resolve() != Path(
            declared["video_path"]
        ).resolve():
            raise ValueError(f"full runtime audit video mismatch for {key}")
        # ``_audit_all_runtimes`` has just hashed every video and compared it
        # with the run_info declaration.  Reuse that verified declaration
        # instead of doing a redundant second nine-video I/O pass.  The one
        # target video is hashed once more immediately before decode below.
        row["video_sha256"] = declared["declared_video_sha256"]
    return audited


def _iter_baseline_reference_paths(
    design: Mapping[str, Any], prompt_id: int
) -> Iterable[Path]:
    prereg = design["baseline_prereg"]
    comparators = prereg.get("comparators")
    if not isinstance(comparators, Mapping):
        raise ValueError("baseline preregistration has no comparators")
    for comparator in ("rvq_quantvideo_int2", "uniform_structured_b2"):
        record = comparators.get(comparator)
        if not isinstance(record, Mapping):
            raise ValueError(f"baseline comparator {comparator} is missing")
        similarity_path = Path(
            str(record.get("similarity_json", ""))
        ).resolve()
        payload = _load_json(similarity_path)
        dev_gate._validate_similarity_protocol(
            payload, source=similarity_path
        )
        videos = payload.get("videos")
        if not isinstance(videos, list):
            raise ValueError(f"{similarity_path}: videos list is missing")
        selected = [
            row for row in videos if int(row.get("prompt_id", -1)) == prompt_id
        ]
        if len(selected) != 1:
            raise ValueError(
                f"{similarity_path}: prompt {prompt_id} reference is ambiguous"
            )
        yield Path(str(selected[0].get("reference_video", ""))).resolve()


def resolve_reference(
    design: Mapping[str, Any], prompt_id: int
) -> dict[str, Any]:
    """Resolve and hash one reference only after the candidate runtime gate."""

    frozen = BF16_REFERENCES[prompt_id]
    path = Path(frozen["path"]).resolve()
    baseline_paths = list(_iter_baseline_reference_paths(design, prompt_id))
    if any(candidate != path for candidate in baseline_paths):
        raise ValueError(
            f"prompt {prompt_id}: comparator BF16 reference path changed"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != int(frozen["bytes"]):
        raise ValueError(
            f"{path}: BF16 reference size changed: {actual_bytes}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != frozen["sha256"]:
        raise ValueError(f"{path}: frozen BF16 reference hash changed")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "bytes": actual_bytes,
        "confirmed_by_comparators": [
            "rvq_quantvideo_int2",
            "uniform_structured_b2",
        ],
    }


def evaluate_video(
    reference_path: Path,
    candidate_path: Path,
    *,
    device_kind: str,
    gpu: int,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Use the frozen evaluator implementation with normalized LPIPS-v2."""

    if sha256_file(EVALUATOR_PATH) != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("frozen compare_video_quality.py hash changed")

    # Heavy imports and CUDA initialization intentionally occur only here,
    # after the complete runtime barrier and input-hash validation.
    import torch
    from scripts.eval import compare_video_quality as quality

    if device_kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if gpu < 0 or gpu >= torch.cuda.device_count():
            raise ValueError(
                f"GPU index {gpu} is outside visible range "
                f"[0,{torch.cuda.device_count() - 1}]"
            )
        device = torch.device(f"cuda:{gpu}")
        gpu_type = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        gpu_type = "CPU"

    model = quality.lpips.LPIPS(net="vgg").to(device).eval()
    result = quality.compare_pair(
        reference_path,
        candidate_path,
        model,
        device,
        include_first_frame=False,
        lpips_normalize_input=True,
    )
    protocol = quality.metric_protocol(
        model,
        include_first_frame=False,
        lpips_normalize_input=True,
    )
    if protocol.get("protocol_id") != SIMILARITY_PROTOCOL:
        raise RuntimeError("evaluator did not produce normalized LPIPS-v2")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, protocol, str(device), gpu_type


def _finite_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    typed = {
        "frames": int(result.get("frames", -1)),
        "compared_frames": int(result.get("compared_frames", -1)),
    }
    for metric in ("mse", "psnr", "ssim", "lpips"):
        value = float(result.get(metric, float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"Similarity {metric} is not finite")
        typed[metric] = value
    if typed["frames"] != 173 or typed["compared_frames"] != 172:
        raise ValueError(
            "Similarity frame accounting changed; expected 173/172"
        )
    return typed


def _publish(
    *,
    args: argparse.Namespace,
    unit_dir: Path,
    target_metadata: Mapping[str, Any],
    target_audit: Mapping[str, Any],
    metadata_rows: list[dict[str, Any]],
    reference: Mapping[str, Any],
    candidate_hash: str,
    metrics: Mapping[str, Any],
    protocol: Mapping[str, Any],
    device_used: str,
    gpu_type: str,
    design: Mapping[str, Any],
) -> Path:
    unit_dir.parent.mkdir(parents=True, exist_ok=True)
    unit_dir.mkdir(exist_ok=False)
    output_json = unit_dir / "similarity.json"
    output_csv = unit_dir / "similarity.csv"
    config_path = unit_dir / "experiment_config.json"
    run_info_path = unit_dir / "run_info.json"
    candidate_path = Path(str(target_audit["video_path"])).resolve()
    setting = f"k_layer_{args.candidate}_structured_k_qv2_v"
    row = {
        "prompt_id": args.prompt_id,
        "prompt": target_metadata["prompt"],
        "setting": setting,
        "reference_video": reference["path"],
        "test_video": str(candidate_path),
        **metrics,
        "normalized_lpips_v2": metrics["lpips"],
        "reference_video_sha256": reference["sha256"],
        "test_video_sha256": candidate_hash,
    }
    summary = {
        "schema_version": SIMILARITY_SCHEMA,
        "metric_protocol": dict(protocol),
        "evaluator": {
            "path": str(EVALUATOR_PATH.resolve()),
            "sha256": EXPECTED_EVALUATOR_SHA256,
        },
        "setting": setting,
        "candidate_id": args.candidate,
        "reference": "bf16",
        "test": candidate_path.parent.name,
        "num_videos": 1,
        "mse_mean": metrics["mse"],
        "psnr_mean": metrics["psnr"],
        "ssim_mean": metrics["ssim"],
        "lpips_mean": metrics["lpips"],
        "normalized_lpips_v2_mean": metrics["lpips"],
        "input_video_sha256": {
            "bf16_reference": reference["sha256"],
            "candidate": candidate_hash,
        },
        "runtime_gate": {
            "status": "PASS",
            "metadata_pass_count": len(metadata_rows),
            "full_audit_pass_count": 9,
            "candidate_video_access_after_complete_runtime_gate": True,
        },
        "videos": [row],
    }
    _write_json_create_only(output_json, summary)
    with output_csv.open("x", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "prompt_id",
            "prompt",
            "setting",
            "reference_video",
            "test_video",
            "frames",
            "compared_frames",
            "psnr",
            "ssim",
            "lpips",
            "mse",
            "normalized_lpips_v2",
            "reference_video_sha256",
            "test_video_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: row[field] for field in fieldnames})

    runtime_receipt = [
        {
            key: item[key]
            for key in (
                "candidate_id",
                "prompt_id",
                "run_info_path",
                "run_info_sha256",
                "gate_audit_path",
                "gate_audit_sha256",
                "gate_status",
                "video_path",
                "declared_video_sha256",
            )
        }
        for item in metadata_rows
    ]
    config = {
        "schema_version": SIMILARITY_CONFIG_SCHEMA,
        "command": shlex.join(sys.argv),
        "candidate_id": args.candidate,
        "prompt_id": args.prompt_id,
        "seed": SEED,
        "device_requested": args.device,
        "gpu_requested": args.gpu,
        "device_used": device_used,
        "reference_video": dict(reference),
        "candidate_video": {
            "path": str(candidate_path),
            "sha256": candidate_hash,
        },
        "metric_protocol_id": SIMILARITY_PROTOCOL,
        "include_first_frame": False,
        "lpips_normalize_input": True,
        "design_protocol": str(design["protocol_path"]),
        "design_protocol_sha256": sha256_file(design["protocol_path"]),
        "runtime_gate": {
            "rule": (
                "all nine run_info + generation gate_audit PASS before any "
                "candidate video is hashed/opened"
            ),
            "metadata_receipt": runtime_receipt,
            "target_full_runtime_audit": dict(target_audit),
        },
        "evaluator_path": str(EVALUATOR_PATH.resolve()),
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "output_json": str(output_json.resolve()),
        "output_csv": str(output_csv.resolve()),
    }
    _write_json_create_only(config_path, config)

    target_run_info = _load_json(Path(target_metadata["run_info_path"]))
    run_info = {
        "schema_version": SIMILARITY_RUN_INFO_SCHEMA,
        "git_commit": _git_commit(),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_file(config_path),
        "codebook_path": target_run_info.get("codebook_path"),
        "codebook_hash": target_run_info.get("codebook_hash"),
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type,
        "seed": SEED,
        "prompt_list": [
            {
                "prompt_id": args.prompt_id,
                "prompt": target_metadata["prompt"],
            }
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_json": str(output_json.resolve()),
        "output_json_sha256": sha256_file(output_json),
        "output_csv": str(output_csv.resolve()),
        "output_csv_sha256": sha256_file(output_csv),
        "candidate_video_sha256": candidate_hash,
        "bf16_reference_video_sha256": reference["sha256"],
        "runtime_gate_status": "PASS_9_OF_9_BEFORE_VIDEO_ACCESS",
        "pid": os.getpid(),
    }
    _write_json_create_only(run_info_path, run_info)
    return output_json


def main(argv: list[str] | argparse.Namespace | None = None) -> int:
    args = argv if isinstance(argv, argparse.Namespace) else parse_args(argv)
    unit_dir = (
        args.output_root.expanduser().resolve()
        / args.candidate
        / f"p{args.prompt_id}"
    )
    if unit_dir.exists():
        raise FileExistsError(f"create-only output already exists: {unit_dir}")

    design = dev_gate._validate_design(args.design_dir.expanduser().resolve())
    mappings = _runtime_mappings(args)
    if not args.execute and not args.audit_only:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "candidate_id": args.candidate,
                    "prompt_id": args.prompt_id,
                    "gpu": args.gpu,
                    "runtime_roots": [str(path) for _, path in mappings],
                    "output_json": str(unit_dir / "similarity.json"),
                    "candidate_video_opened": False,
                    "gpu_started": False,
                },
                indent=2,
            )
        )
        return 0

    metadata_rows = metadata_runtime_gate(mappings)
    audited_rows = full_runtime_audit(mappings, design, metadata_rows)
    if args.audit_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "metadata_runtime_audits": len(metadata_rows),
                    "full_runtime_audits": len(audited_rows),
                    "quality_computed": False,
                    "gpu_started": False,
                },
                indent=2,
            )
        )
        return 0

    metadata_by_key = {
        (row["candidate_id"], row["prompt_id"]): row
        for row in metadata_rows
    }
    audited_by_key = {
        (row["candidate_id"], row["prompt_id"]): row
        for row in audited_rows
    }
    key = (args.candidate, args.prompt_id)
    target_metadata = metadata_by_key[key]
    target_audit = audited_by_key[key]
    candidate_path = Path(target_audit["video_path"]).resolve()
    candidate_hash = sha256_file(candidate_path)
    if candidate_hash != target_metadata["declared_video_sha256"]:
        raise ValueError(f"{candidate_path}: candidate video hash changed")
    reference = resolve_reference(design, args.prompt_id)
    result, protocol, device_used, gpu_type = evaluate_video(
        Path(reference["path"]),
        candidate_path,
        device_kind=args.device,
        gpu=args.gpu,
    )
    metrics = _finite_metrics(result)
    output_json = _publish(
        args=args,
        unit_dir=unit_dir,
        target_metadata=target_metadata,
        target_audit=target_audit,
        metadata_rows=metadata_rows,
        reference=reference,
        candidate_hash=candidate_hash,
        metrics=metrics,
        protocol=protocol,
        device_used=device_used,
        gpu_type=gpu_type,
        design=design,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_json": str(output_json),
                "evaluator_argument": (
                    f"--similarity {args.candidate}={output_json}"
                ),
                "mse": metrics["mse"],
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "normalized_lpips_v2": metrics["lpips"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
