#!/usr/bin/env python3
"""Exact, analysis-only oracles for the R²AQ coherent V-tail.

This script intentionally does not define a deployable codec.  It reconstructs
the same prompt/layer capture as the exact R²AQ gate and compares:

* the physical G64 INT4+FP8+uint16 coherent tail;
* FP32 corrections on exactly the same selected groups;
* a dense FP32 correction (coverage upper bound); and
* dense G128 INT4 corrections with one or two FP8 scales.

The oracle rows explicitly distinguish physical payload from counterfactual
payload.  All outputs are create-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parent.parent
SCRIPT_DIR = SCRIPT.parent
for _path in (REPO_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_r2aq_exact_attention as gate  # noqa: E402


DEFAULT_DATA = gate.DEFAULT_DATA
DEFAULT_SOURCE = (
    DEFAULT_DATA
    / "r2aq_coherent_tail_fraction_055_exact_20260724_b4diag_v1"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "r2aq_coherent_tail_oracles_20260724_b4diag_v1"
)
DEFAULT_UTILITY = (
    DEFAULT_DATA
    / "r2aq_shared_params_20260724_root_v1"
    / "utility_weights.json"
)
DEFAULT_TREE = (
    DEFAULT_DATA
    / "r2aq_shared_params_20260724_root_v1"
    / "params.pt"
)
FRACTIONS = (0.25, 0.4, 0.55, 0.7, 1.0)
TAIL_METRICS = gate.TAIL_METRICS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-exact-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--utility-weights", type=Path, default=DEFAULT_UTILITY)
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fractions", default=",".join(map(str, FRACTIONS)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260724)
    args = parser.parse_args(argv)
    for name in (
        "data_root",
        "source_exact_root",
        "output_root",
        "utility_weights",
        "tree_params",
    ):
        setattr(args, name, getattr(args, name).resolve())
    try:
        args.fractions = tuple(
            float(item.strip())
            for item in str(args.fractions).split(",")
            if item.strip()
        )
    except ValueError as error:
        parser.error(f"invalid --fractions: {error}")
    if not args.fractions or any(
        not 0.0 <= value <= 1.0 for value in args.fractions
    ):
        parser.error("--fractions must contain values in [0,1]")
    if args.output_root.exists():
        parser.error(f"create-only output already exists: {args.output_root}")
    if not (args.source_exact_root / "config.json").is_file():
        parser.error("--source-exact-root must contain config.json")
    if not args.utility_weights.is_file() or not args.tree_params.is_file():
        parser.error("shared utility/tree parameter files are required")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("oracle exact replay requires CUDA")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    fields = sorted({key for row in rows for key in row})
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def cluster_means(
    error: torch.Tensor,
    cluster_ids: torch.Tensor,
    n_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, _tokens, dim = error.shape
    if batch != 1:
        raise ValueError("oracle requires batch size 1")
    means = torch.zeros(
        heads,
        n_clusters,
        dim,
        dtype=torch.float32,
        device=error.device,
    )
    counts = torch.zeros(
        heads, n_clusters, dtype=torch.float32, device=error.device
    )
    ids = cluster_ids[0].long()
    for head in range(heads):
        head_ids = ids[head]
        counts[head].scatter_add_(
            0, head_ids, torch.ones_like(head_ids, dtype=torch.float32)
        )
        means[head].scatter_add_(
            0,
            head_ids[:, None].expand(-1, dim),
            error[0, head],
        )
    means /= counts.clamp_min(1).unsqueeze(-1)
    return means, counts


def gather_correction(
    table: torch.Tensor,
    cluster_ids: torch.Tensor,
) -> torch.Tensor:
    batch, heads, tokens = cluster_ids.shape
    dim = int(table.shape[-1])
    return torch.gather(
        table.unsqueeze(0),
        2,
        cluster_ids.long().unsqueeze(-1).expand(
            batch, heads, tokens, dim
        ),
    )


def selected_fp32_table(
    means: torch.Tensor,
    group_ids: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    heads, clusters, dim = means.shape
    blocks = means.reshape(
        heads * clusters * (dim // group_size), group_size
    )
    result = torch.zeros_like(blocks)
    result[group_ids.long().bitwise_and(0xFFFF)] = blocks[
        group_ids.long().bitwise_and(0xFFFF)
    ]
    return result.reshape(heads, clusters, dim)


def quantize_g128(
    module: Any,
    means: torch.Tensor,
    scales_per_vector: int,
) -> torch.Tensor:
    if scales_per_vector not in (1, 2):
        raise ValueError("scales_per_vector must be 1 or 2")
    group = 128 // scales_per_vector
    blocks = means.reshape(-1, group)
    raw = module._to_positive_e4m3_raw(  # noqa: SLF001
        blocks.abs().amax(-1, keepdim=True).div(7)
    )
    scale = module._from_positive_e4m3_raw(raw)  # noqa: SLF001
    safe = scale.clamp_min(torch.finfo(torch.float32).tiny)
    values = torch.round(blocks / safe).clamp(-7, 7)
    values = torch.where(scale > 0, values, torch.zeros_like(values))
    return (values * scale).reshape_as(means)


def metric_rows(
    *,
    fraction: float,
    variant: str,
    reconstructed_k: torch.Tensor,
    reconstructed_v: torch.Tensor,
    value_reference: torch.Tensor,
    args: Any,
    tensors: Mapping[str, torch.Tensor],
    evidence: Any,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    query_output_matrix: torch.Tensor | None,
    history_tokens: int,
    halves: Mapping[str, tuple[torch.Tensor, ...]],
    reference_context: Mapping[str, Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metrics, _current, _candidates = gate.score_exact_attention(
        args=args,
        tensors=tensors,
        evidence=evidence,
        query_positions=query_positions,
        key_positions=key_positions,
        query_output_matrix=query_output_matrix,
        history_tokens=history_tokens,
        reconstructed_k=reconstructed_k,
        reconstructed_v=reconstructed_v,
        halves=halves,
        reference_context=reference_context,
    )
    v_mse = float(
        (reconstructed_v.float() - value_reference.float()).square().mean()
    )
    rows: list[dict[str, Any]] = []
    for half in gate.ATTENTION_HALVES:
        for intervention in gate.INTERVENTIONS:
            head_rows = metrics[half][intervention]
            row = {
                "fraction": fraction,
                "variant": variant,
                "attention_half": half,
                "intervention": intervention,
                "historical_V_mse": v_mse,
                **payload,
            }
            for name in TAIL_METRICS:
                row[name] = float(
                    np.mean([float(item[name]) for item in head_rows])
                )
            rows.append(row)
    return rows


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    source_config_path = args.source_exact_root / "config.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    captures = source_config.get("captures", [])
    if len(captures) != 1:
        raise ValueError("oracle expects exactly one exact capture")
    capture_info = captures[0]
    capture_path = Path(capture_info["capture_path"])
    if sha256_file(capture_path) != capture_info["capture_sha256"]:
        raise RuntimeError("capture hash mismatch")
    prompt_id = int(capture_info["prompt_id"])
    layer = int(capture_info["layer"])
    prompt = str(capture_info["prompt"])
    device = torch.device(args.device)

    exact_args = gate.parse_args(
        [
            "create",
            "--data-root",
            str(args.data_root),
            "--output-root",
            str(args.output_root / "_unused"),
            "--prompts",
            str(prompt_id),
            "--layers",
            str(layer),
            "--ablations",
            "coherent_tail",
            "--device",
            args.device,
            "--utility-weights",
            str(args.utility_weights),
            "--tree-params",
            str(args.tree_params),
            "--seed",
            str(args.seed),
            "--codec-seed",
            str(args.codec_seed),
            "--no-metric-audit",
        ]
    )
    task = {
        "prompt_id": prompt_id,
        "layer": layer,
        "prompt": prompt,
        "task_index": 0,
    }

    raw = torch.load(
        str(capture_path), map_location="cpu", weights_only=False, mmap=True
    )
    metadata = gate.replay.exact.validate_capture(raw)
    (
        tensors,
        evidence,
        positional_validation,
        query_output_matrix,
        _full_kv_matrix,
        query_positions,
        key_positions,
    ) = gate.replay.exact.positional_replay(raw, device)
    del raw
    history_tokens = int(metadata["cache_tokens"])
    key_reference = tensors["key_pre"][:, :, :history_tokens]
    value_reference = tensors["value_pre"][:, :, :history_tokens]
    references = {"K": key_reference, "V": value_reference}
    partition = gate.runtime_history_partition(history_tokens)
    spans = tuple(partition["quantized_spans"])
    pending = partition["pending_bf16_range"]
    codebook_dirs = {
        "K": args.data_root / "nonorm_k_l8c64_codebooks",
        "V": args.data_root / "nonorm_v_l8c64_codebooks",
    }
    for name, directory in codebook_dirs.items():
        evidence.validate_codebook_dir(
            directory,
            layer=layer,
            heads=gate.replay.p0.HEADS,
            expected_dim=gate.replay.p0.DIM,
        )

    bases: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    residuals: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    for name in ("K", "V"):
        for start, end in spans:
            source = references[name][:, :, start:end].contiguous()
            packed = evidence.encode_fixed(
                source,
                layer=layer,
                directory=codebook_dirs[name],
                batch_size=exact_args.assignment_batch_size,
            )
            base = evidence.rvq_dequantize_tensor_cache(
                packed, torch.bfloat16
            )
            bases[name].append(base)
            residuals[name].append((source - base).contiguous())

    halves = {
        "standard_rope": (
            tensors["query_post_rope"],
            tensors["key_post_rope"],
            tensors["value_post_rope"],
            None,
        ),
        "camera_prope": (
            tensors["query_post_prope"],
            tensors["key_post_prope"],
            tensors["value_post_prope"],
            query_output_matrix,
        ),
    }
    reference_context = {
        half: gate.replay.build_reference_context(
            query=query,
            key=key,
            value=value,
            history_tokens=history_tokens,
            query_positions=query_positions,
            key_positions=key_positions,
            query_batch_size=exact_args.query_batch_size,
            topk=exact_args.topk,
            output_matrix=matrix,
            evidence=evidence,
        )
        for half, (query, key, value, matrix) in halves.items()
    }

    module = gate.codec_module()
    k_parts: list[torch.Tensor] = []
    k_seed = gate.protocol.method_seed(
        exact_args, task, "r2aq_coherent_tail", 4, "K"
    )
    for span_index, residual in enumerate(residuals["K"]):
        ages = torch.full(
            (residual.shape[2],),
            min(span_index, exact_args.age_bins - 1),
            dtype=torch.long,
            device=device,
        )
        with gate.protocol.deterministic_rng(device, k_seed):
            state = gate.encode_r2aq(
                module,
                residual,
                tensor_name="K",
                ablation="coherent_tail",
                layer=layer,
                ages=ages,
                args=exact_args,
            )
        decoded = module.decode_r2aq_residual(
            state, output_dtype=torch.float32
        )
        k_parts.append(
            (bases["K"][span_index] + decoded)
            .to(torch.bfloat16)
            .contiguous()
        )
    if pending is not None:
        k_parts.append(key_reference[:, :, pending[0] : pending[1]])
    reconstructed_k = torch.cat(k_parts, dim=2)

    rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    for fraction in args.fractions:
        exact_args.coherent_tail_group_fraction = fraction
        real_parts: list[torch.Tensor] = []
        selected_fp32_parts: list[torch.Tensor] = []
        dense_fp32_parts: list[torch.Tensor] = []
        g128_one_parts: list[torch.Tensor] = []
        g128_two_parts: list[torch.Tensor] = []
        selected_groups = 0
        possible_groups = 0
        physical_tail_bytes = 0
        local_payload_bytes = 0
        local_selected_fractions: list[float] = []
        v_seed = gate.protocol.method_seed(
            exact_args, task, "r2aq_coherent_tail", 4, "V"
        )
        for span_index, residual in enumerate(residuals["V"]):
            ages = torch.full(
                (residual.shape[2],),
                min(span_index, exact_args.age_bins - 1),
                dtype=torch.long,
                device=device,
            )
            with gate.protocol.deterministic_rng(device, v_seed):
                state = gate.encode_r2aq(
                    module,
                    residual,
                    tensor_name="V",
                    ablation="coherent_tail",
                    layer=layer,
                    ages=ages,
                    args=exact_args,
                )
            real = module.decode_r2aq_residual(
                state, output_dtype=torch.float32
            )
            tail = module._decode_coherent_tail(state)  # noqa: SLF001
            local = real - tail
            error = residual.float() - local.float()
            ids = state["predictor"]["cluster_ids"]
            means, _counts = cluster_means(
                error, ids, int(state["n_clusters"])
            )
            chosen = state["coherent_tail"]["group_ids"]
            selected_table = selected_fp32_table(means, chosen)
            dense_table = means
            g128_one = quantize_g128(module, means, 1)
            g128_two = quantize_g128(module, means, 2)
            candidates = (
                (real_parts, real),
                (
                    selected_fp32_parts,
                    local + gather_correction(selected_table, ids),
                ),
                (
                    dense_fp32_parts,
                    local + gather_correction(dense_table, ids),
                ),
                (
                    g128_one_parts,
                    local + gather_correction(g128_one, ids),
                ),
                (
                    g128_two_parts,
                    local + gather_correction(g128_two, ids),
                ),
            )
            for destination, decoded in candidates:
                destination.append(
                    (bases["V"][span_index] + decoded)
                    .to(torch.bfloat16)
                    .contiguous()
                )
            count = int(state["coherent_tail"]["selected_group_count"])
            selected_groups += count
            possible_groups += (
                int(state["shape"][1])
                * int(state["n_clusters"])
                * int(state["coherent_tail"]["channel_groups"])
            )
            physical_tail_bytes += (
                int(state["coherent_tail"]["packed_values"].numel())
                + int(state["coherent_tail"]["scale_raw"].numel())
                + int(state["coherent_tail"]["group_ids"].numel())
                * state["coherent_tail"]["group_ids"].element_size()
            )
            local_payload_bytes += int(module.payload_nbytes(state))
            local_selected_fractions.append(
                float(state["realized_selection_fraction"])
            )
        if pending is not None:
            pending_v = value_reference[:, :, pending[0] : pending[1]]
            for parts in (
                real_parts,
                selected_fp32_parts,
                dense_fp32_parts,
                g128_one_parts,
                g128_two_parts,
            ):
                parts.append(pending_v)

        centroids = len(spans) * gate.replay.p0.HEADS * 256
        dense_g64_wire = centroids * 2 * (32 + 1 + 2)
        dense_g128_one_wire = centroids * (64 + 1 + 2)
        dense_g128_two_wire = centroids * (64 + 2 + 2)
        shared = {
            "requested_fraction": fraction,
            "actual_tail_group_count": selected_groups,
            "possible_tail_group_count": possible_groups,
            "actual_tail_group_fraction": (
                selected_groups / possible_groups
                if possible_groups
                else 0.0
            ),
            "local_refinement_selected_fraction": float(
                np.mean(local_selected_fractions)
            ),
            "physical_g64_tail_bytes": physical_tail_bytes,
            "r2aq_local_payload_bytes": local_payload_bytes,
            "dense_g64_wire_bytes": dense_g64_wire,
            "dense_g128_one_scale_wire_bytes": dense_g128_one_wire,
            "dense_g128_two_scale_wire_bytes": dense_g128_two_wire,
            "dense_g128_one_scale_saving_bytes_vs_g64": (
                dense_g64_wire - dense_g128_one_wire
            ),
            "dense_g128_two_scale_saving_bytes_vs_g64": (
                dense_g64_wire - dense_g128_two_wire
            ),
        }
        payload_rows.append(dict(shared))
        variants = (
            ("real_int4_g64", real_parts, True, physical_tail_bytes),
            (
                "oracle_fp32_same_selected_groups",
                selected_fp32_parts,
                False,
                selected_groups * (64 * 4 + 2),
            ),
            (
                "oracle_fp32_dense_maxcount",
                dense_fp32_parts,
                False,
                possible_groups * 64 * 4,
            ),
            (
                "diagnostic_int4_g128_one_scale_dense",
                g128_one_parts,
                dense_g128_one_wire <= physical_tail_bytes,
                dense_g128_one_wire,
            ),
            (
                "diagnostic_int4_g128_two_scale_dense",
                g128_two_parts,
                dense_g128_two_wire <= physical_tail_bytes,
                dense_g128_two_wire,
            ),
        )
        for variant, parts, feasible, diagnostic_bytes in variants:
            payload = {
                **shared,
                "physical_codec": variant == "real_int4_g64",
                "counterfactual_tail_bytes": diagnostic_bytes,
                "fits_current_fraction_tail_reservation": feasible,
                "payload_interpretation": (
                    "strict_physical_equal_rate"
                    if variant == "real_int4_g64"
                    else "diagnostic_counterfactual_not_a_wire"
                ),
            }
            rows.extend(
                metric_rows(
                    fraction=fraction,
                    variant=variant,
                    reconstructed_k=reconstructed_k,
                    reconstructed_v=torch.cat(parts, dim=2),
                    value_reference=value_reference,
                    args=exact_args,
                    tensors=tensors,
                    evidence=evidence,
                    query_positions=query_positions,
                    key_positions=key_positions,
                    query_output_matrix=query_output_matrix,
                    history_tokens=history_tokens,
                    halves=halves,
                    reference_context=reference_context,
                    payload=payload,
                )
            )

    config = {
        "schema_version": "r2aq_coherent_tail_oracle.config.v1",
        "analysis_only": True,
        "source_exact_root": str(args.source_exact_root),
        "source_exact_config_hash": sha256_file(source_config_path),
        "capture_path": str(capture_path),
        "capture_hash": capture_info["capture_sha256"],
        "prompt_id": prompt_id,
        "prompt": prompt,
        "layer": layer,
        "fractions": list(args.fractions),
        "device": args.device,
        "utility_weights": str(args.utility_weights),
        "utility_weights_hash": sha256_file(args.utility_weights),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "metric_scope": gate.EXACT_ATTENTION_SCOPE,
        "oracle_definitions": {
            "same_selected": "FP32 cluster mean on physical G64 selected IDs",
            "dense_maxcount": "FP32 cluster mean on every G64 group",
            "g128_one_scale": "dense INT4 vector with one FP8 scale and uint16 ID",
            "g128_two_scale": "dense INT4 vector with two FP8 half-scales and uint16 ID",
        },
        "positional_validation": positional_validation,
    }
    config_payload = json_bytes(config)
    config_hash = hashlib.sha256(config_payload).hexdigest()
    outputs = {
        "config.json": config_payload,
        "oracle_metrics.csv": csv_bytes(rows),
        "oracle_payload.csv": csv_bytes(payload_rows),
    }
    args.output_root.mkdir(parents=True)
    for name, payload in outputs.items():
        create_only(args.output_root / name, payload)
    gpu_name = torch.cuda.get_device_name(device)
    run_info = {
        "schema_version": "r2aq_coherent_tail_oracle.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(args.output_root / "config.json"),
        "config_hash": config_hash,
        "codebook_path": {
            name: str(path) for name, path in codebook_dirs.items()
        },
        "codebook_hash": source_config.get("codebook_hash"),
        "hostname": socket.gethostname(),
        "gpu_type": gpu_name,
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "prompt_list": [prompt],
        "timestamp": utc_now(),
        "source_hashes": {
            str(SCRIPT): sha256_file(SCRIPT),
            str(REPO_ROOT / "quant_videogen" / "r2aq.py"): sha256_file(
                REPO_ROOT / "quant_videogen" / "r2aq.py"
            ),
            str(SCRIPT_DIR / "evaluate_r2aq_exact_attention.py"): sha256_file(
                SCRIPT_DIR / "evaluate_r2aq_exact_attention.py"
            ),
        },
        "output_hashes": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in outputs.items()
        },
    }
    create_only(args.output_root / "run_info.json", json_bytes(run_info))
    print(
        f"[done] rows={len(rows)} fractions={len(args.fractions)} "
        f"output={args.output_root}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
