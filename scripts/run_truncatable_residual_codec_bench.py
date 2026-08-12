#!/usr/bin/env python3
"""Train and evaluate truncatable codecs for an L8C64-INT4 RVQ residual.

The formal offline path is deliberately separate from video generation.  It
uses the fixed 20-train/10-test Wan2.2-5B BF16 KV dump and evaluates attention
on a deterministic sampled inference span.  These attention numbers are an
offline proxy: they do not replay positional encoding or the complete online
attention kernel.

Examples (four GPUs):

  python scripts/run_truncatable_residual_codec_bench.py train
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/run_truncatable_residual_codec_bench.py \
      offline --shard-index $i --num-shards 4 &
  done
  python scripts/run_truncatable_residual_codec_bench.py aggregate
  python scripts/run_truncatable_residual_codec_bench.py benchmark

Every mutating command refuses to overwrite an existing artifact.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_fixed_budget_rvq_screen import (  # noqa: E402
    baseline_path,
    layer_path,
    nearest,
    parse_ints,
    quantize_codebook_int4,
    sha256_file,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = DEFAULT_DATA / "truncatable_residual_benchmark_20260722"
SCHEMES = ("nested_affine", "gaussian_tree", "additive_binary", "hierarchical_vq")
TENSOR_NAMES = ("K", "V")
DIM = 128
MAX_BITS = 4
PREFIX_WEIGHTS = (0.1, 1.0, 0.2, 1.0)
RVQ_ASSIGNMENT_MODES = ("runtime_bf16", "fp32_proxy")
GROUP_SIZE = {
    "nested_affine": 32,
    "gaussian_tree": 64,
    "additive_binary": 64,
    "hierarchical_vq": 64,
}
FORMAL_TEST_PROMPTS = tuple(range(10))
FORMAL_LAYERS = tuple(range(30))
FORMAL_HEADS = tuple(range(24))
RECONSTRUCTION_PROTOCOL_BY_ASSIGNMENT = {
    "runtime_bf16": (
        "runtime_bf16_triton_int4_bf16_codebook_fp16_scale_"
        "stagewise_bf16_decode_subtract"
    ),
    "fp32_proxy": (
        "fp32_assignment_accumulation_proxy_with_bf16_final_base_subtract"
    ),
}
# Default protocol retained as a convenience for validation helpers/tests.  It
# deliberately does not accept the legacy "bfloat16_runtime_faithful" label:
# that path assigned and accumulated in FP32 and was therefore only a proxy.
OFFLINE_RECONSTRUCTION_PROTOCOL = RECONSTRUCTION_PROTOCOL_BY_ASSIGNMENT[
    "runtime_bf16"
]
OFFLINE_METRIC_NAMES = (
    "K_MSE",
    "V_MSE",
    "K_relative_MSE",
    "V_relative_MSE",
    "QK_logits_MSE",
    "attention_probability_MSE",
    "attention_KL",
    "QKV_output_MSE",
    "QKV_output_cosine",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("command", choices=("train", "select", "offline", "benchmark", "aggregate"))
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--schemes", default=",".join(SCHEMES))
    p.add_argument("--train-prompts", default="0:20")
    p.add_argument("--test-prompts", default="0:10")
    p.add_argument("--layers", default="0:30")
    p.add_argument("--heads", default="0:24")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--codec-seed", type=int, default=20260722)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--rvq-assignment-mode",
        choices=RVQ_ASSIGNMENT_MODES,
        default="runtime_bf16",
        help=(
            "runtime_bf16 exactly uses the deployment Triton assignment/decode "
            "path; fp32_proxy preserves the old analytical proxy and is never "
            "reported as runtime-faithful"
        ),
    )

    p.add_argument("--train-samples", type=int, default=400_000,
                   help="Balanced residual vectors retained per K and V.")
    p.add_argument("--train-units-per-prompt", type=int, default=2)
    p.add_argument("--train-eval-samples", type=int, default=65_536)
    p.add_argument("--tree-iters", type=int, default=8)
    p.add_argument("--additive-iters", type=int, default=3)
    p.add_argument("--hvq-iters", type=int, default=3)
    p.add_argument("--hvq-train-vectors", type=int, default=200_000,
                   help="Maximum normalized 4-D subvectors per K/V and transform.")

    p.add_argument("--key-sample", type=int, default=7040)
    p.add_argument("--query-sample", type=int, default=96)
    p.add_argument("--key-sampling", choices=("inference_span", "random"), default="inference_span")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--bootstrap-samples", type=int, default=10_000)

    p.add_argument("--benchmark-prompt", type=int, default=0)
    p.add_argument("--benchmark-layer", type=int, default=0)
    p.add_argument("--benchmark-warmup", type=int, default=5)
    p.add_argument("--benchmark-trials", type=int, default=20)
    args = p.parse_args()
    args.schemes = tuple(x.strip() for x in args.schemes.split(",") if x.strip())
    unknown = sorted(set(args.schemes) - set(SCHEMES))
    if unknown:
        p.error(f"unknown schemes: {unknown}")
    if not 0 <= args.shard_index < args.num_shards:
        p.error("--shard-index must be in [0, --num-shards)")
    if args.bootstrap_samples < 1:
        p.error("--bootstrap-samples must be positive")
    return args


def codec_module() -> Any:
    from quant_videogen import truncatable_residual
    return truncatable_residual


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def refuse_existing(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing artifact(s): " + ", ".join(existing))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def stable_directory_hash(path: Path) -> str:
    """Hash all file contents and relative names in deterministic order."""
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(b"\0")
        with item.open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
    return h.hexdigest()


def tensor_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_nbytes(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_nbytes(x) for x in value)
    return 0


def split_payload(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads((args.data_root / "prompt_split.json").read_text(encoding="utf-8"))


def prompt_texts(args: argparse.Namespace, split_name: str, ids: Iterable[int]) -> list[str]:
    items = split_payload(args)[split_name]
    by_id = {int(x["prompt_id"]): x["prompt"] for x in items}
    return [by_id[i] for i in ids]


def device_from_args(args: argparse.Namespace) -> torch.device:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(args.device)


def assignment_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "rvq_assignment_mode", "runtime_bf16")).lower()
    if mode not in RVQ_ASSIGNMENT_MODES:
        raise ValueError(
            f"Unknown RVQ assignment mode {mode!r}; expected {RVQ_ASSIGNMENT_MODES}"
        )
    return mode


def reconstruction_protocol(args: argparse.Namespace) -> str:
    return RECONSTRUCTION_PROTOCOL_BY_ASSIGNMENT[assignment_mode(args)]


def rvq_runtime_environment() -> dict[str, Any]:
    names = (
        "HYWORLDPLAY_RVQ_TRITON_FASTPATH",
        "HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE",
        "HYWORLDPLAY_RVQ_NUM_LEVELS",
        "HYWORLDPLAY_RVQ_TRITON_BLOCK_TOKENS",
        "HYWORLDPLAY_RVQ_TRITON_DECODE_BLOCK_TOKENS",
    )
    return {
        "raw": {name: os.getenv(name) for name in names},
        "effective_triton_fastpath": (
            os.getenv("HYWORLDPLAY_RVQ_TRITON_FASTPATH", "1").lower()
            not in {"0", "false", "no", "off"}
        ),
        "effective_runtime_codebook_cache": (
            os.getenv("HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE", "1").lower()
            not in {"0", "false", "no", "off"}
        ),
    }


def assignment_protocol_metadata(args: argparse.Namespace) -> dict[str, Any]:
    mode = assignment_mode(args)
    if mode == "runtime_bf16":
        return {
            "mode": mode,
            "runtime_faithful": True,
            "raw_codebook_cast_before_int4": "bfloat16",
            "int4_scale_dtype": "float16",
            "assignment": "quant_videogen.kernels.triton.rvq.rvq_encode_bhsd_triton",
            "assignment_dot_rounding": "BF16 inputs; Triton dot rounded to BF16 before distance",
            "residual_update_rounding": "bfloat16 after every RVQ level",
            "decode_accumulation_rounding": "bfloat16 after every RVQ level",
            "final_residual_subtraction": "BF16 reference minus BF16 RVQ decode",
            "runtime_environment": rvq_runtime_environment(),
            "reconstruction_protocol": reconstruction_protocol(args),
        }
    return {
        "mode": mode,
        "runtime_faithful": False,
        "warning": (
            "Analytical legacy proxy: INT4-dequantized codebooks, assignment, "
            "residual updates, and base accumulation are FP32; only the final "
            "base cast and subtraction are BF16."
        ),
        "runtime_environment": rvq_runtime_environment(),
        "reconstruction_protocol": reconstruction_protocol(args),
    }


def implementation_source_hashes() -> dict[str, str]:
    return {
        "offline_benchmark_script_sha256": sha256_file(Path(__file__)),
        "truncatable_residual_module_sha256": sha256_file(
            REPO_ROOT / "quant_videogen" / "truncatable_residual.py"
        ),
        "rvq_runtime_module_sha256": sha256_file(
            REPO_ROOT / "quant_videogen" / "rvq.py"
        ),
        "rvq_triton_kernel_sha256": sha256_file(
            REPO_ROOT / "quant_videogen" / "kernels" / "triton" / "rvq.py"
        ),
        "compress_module_sha256": sha256_file(
            REPO_ROOT / "quant_videogen" / "compress.py"
        ),
        "uncompress_module_sha256": sha256_file(
            REPO_ROOT / "quant_videogen" / "uncompress.py"
        ),
        "fixed_budget_rvq_helper_sha256": sha256_file(
            REPO_ROOT / "scripts" / "run_fixed_budget_rvq_screen.py"
        ),
    }


def config_dict(args: argparse.Namespace, command: str) -> dict[str, Any]:
    raw = vars(args).copy()
    raw["data_root"] = str(args.data_root)
    raw["output_root"] = str(args.output_root)
    raw["schemes"] = list(args.schemes)
    raw["command"] = command
    raw["experiment_protocol_version"] = "truncatable_residual_strict_v2"
    raw["rvq_assignment"] = assignment_protocol_metadata(args)
    raw["implementation_source_hash"] = implementation_source_hashes()
    raw["metric_accounting"] = {
        "memory_unit": "MiB (2**20 bytes)",
        "compression_baseline": "BF16 residual, 16 bits/value; RVQ payload excluded from residual ratio",
        "residual_payload": (
            "packed physical bitplanes plus FP16 per-block min/max or RMS scale; "
            "Python/container headers and globally shared codec tables excluded"
        ),
        "shared_parameter_accounting": (
            "selected K/V tables reported separately; charged once (not once per "
            "layer/head/prompt) in dataset-level ratios"
        ),
        "K_V_errors": "reported separately",
        "attention_scope": (
            "offline sampled-attention proxy on one deterministic inference-span; "
            "no positional replay and not the online attention kernel"
        ),
        "reconstruction_dtype": (
            f"{reconstruction_protocol(args)}; residual codec decode is BF16, "
            "then added to the BF16 RVQ base before comparison/attention"
        ),
        "runtime_interval": (
            "residual codec only; B4 encode, state prefix truncation, and each prefix decode; "
            "L8C64 RVQ reconstruction excluded"
        ),
    }
    return raw


def write_run_info(
    args: argparse.Namespace,
    command: str,
    config_path: Path,
    run_info_path: Path,
    prompt_ids: list[int],
    split_name: str,
    *,
    config_extra: dict[str, Any] | None = None,
    info_extra: dict[str, Any] | None = None,
) -> None:
    refuse_existing(config_path, run_info_path)
    cfg = config_dict(args, command)
    if config_extra:
        cfg.update(config_extra)
    atomic_json(config_path, cfg)
    cb_dirs = {
        "K": args.data_root / "nonorm_k_l8c64_codebooks",
        "V": args.data_root / "nonorm_v_l8c64_codebooks",
    }
    registered = {
        "K": "5338f0089646aaeaf4ddce7a7ebef29d0185ba3bbe636cdf95746e1056a4d828",
        "V": "394d375d7d1440df92741517ff62e99fbf0da918e76c0a579f3608d2cbc611b6",
    }
    actual_codebook_hash = {
        key: stable_directory_hash(path) for key, path in cb_dirs.items()
    }
    training_manifest_path = (
        args.output_root / "training" / "codec_params_manifest.json"
    )
    selection_path = args.output_root / "training" / "selection.json"
    info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": {k: str(v) for k, v in cb_dirs.items()},
        "codebook_hash": actual_codebook_hash,
        "registered_dataset_codebook_hash": registered,
        "codebook_marker_hash": {
            k: sha256_file(v / "layer_0_head_0.pt") for k, v in cb_dirs.items()
        },
        "codec_params_path": str(args.output_root / "training" / "codec_params.pt"),
        "codec_params_hash": (
            sha256_file(args.output_root / "training" / "codec_params.pt")
            if (args.output_root / "training" / "codec_params.pt").exists() else None
        ),
        "codec_params_manifest_path": str(training_manifest_path),
        "codec_params_manifest_hash": (
            sha256_file(training_manifest_path)
            if training_manifest_path.exists() else None
        ),
        "selection_path": str(selection_path),
        "selection_hash": (
            sha256_file(selection_path) if selection_path.exists() else None
        ),
        "implementation_hash": implementation_source_hashes(),
        "experiment_protocol_version": "truncatable_residual_strict_v2",
        "rvq_assignment": assignment_protocol_metadata(args),
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available() else None
        ),
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "prompt_list": prompt_texts(args, split_name, prompt_ids),
        "prompt_ids": prompt_ids,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    if info_extra:
        info.update(info_extra)
    atomic_json(run_info_path, info)


def runtime_int4_codebook_bf16(
    codebooks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize deployment INT4 codebooks after the mandatory BF16 cast.

    The order is material: the runtime loader casts checkpoint centroids to
    BF16 *before* deriving one FP16 scale per centroid.  Returning the scale as
    well makes the precision contract directly testable and auditable.
    """

    bf16 = codebooks.to(dtype=torch.bfloat16)
    max_abs = bf16.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (max_abs / 7.0).clamp_min(torch.finfo(torch.float16).tiny).to(
        torch.float16
    )
    q_signed = torch.round(bf16.float() / scale.float()).clamp(-7, 7)
    runtime = (
        q_signed.to(torch.bfloat16) * scale.to(torch.bfloat16)
    ).contiguous()
    return runtime, scale.contiguous()


@functools.lru_cache(maxsize=5760)
def _cached_int4_codebook(
    data_root: str,
    tensor_name: str,
    layer: int,
    head: int,
    device_name: str,
    mode: str,
) -> torch.Tensor:
    path = baseline_path(Path(data_root), "l8", tensor_name, layer, head)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    input_mode = str(
        payload.get(
            "input_norm_mode",
            payload.get("config", {}).get("input_norm_mode", "none"),
        )
    ).lower()
    if input_mode not in ("none", "raw"):
        raise ValueError(
            f"Expected raw/no-norm L8C64 codebook, got {input_mode}: {path}"
        )
    if mode == "runtime_bf16":
        runtime, _scale = runtime_int4_codebook_bf16(
            payload["codebooks"].to(device=device_name, dtype=torch.bfloat16)
        )
        return runtime
    if mode == "fp32_proxy":
        # Preserve the historical proxy byte-for-byte: quantize checkpoint
        # FP32 centroids with FP16 scales on CPU, then move the FP32 table.
        return quantize_codebook_int4(payload["codebooks"].float()).to(device_name)
    raise ValueError(f"Unsupported RVQ assignment mode: {mode}")


def load_int4_base_codebook(
    args: argparse.Namespace, tensor_name: str, layer: int, head: int, device: torch.device
) -> torch.Tensor:
    return _cached_int4_codebook(
        str(args.data_root.resolve()),
        tensor_name,
        int(layer),
        int(head),
        str(device),
        assignment_mode(args),
    )


def fp32_proxy_residual(
    x: torch.Tensor, codebook: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the historical FP32-assignment proxy (not runtime-faithful)."""

    residual_fp32 = x.to(device=device, dtype=torch.float32).clone()
    base = torch.zeros_like(residual_fp32)
    for level in codebook[:8]:
        ids = nearest(residual_fp32, level, device)
        chosen = level[ids]
        base.add_(chosen)
        residual_fp32.sub_(chosen)
    ref_bf16 = x.to(device=device, dtype=torch.bfloat16)
    base_bf16 = base.to(dtype=torch.bfloat16)
    residual = (ref_bf16 - base_bf16).to(torch.float32)
    return base_bf16, residual


def runtime_bf16_residual(
    x: torch.Tensor, codebook: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the no-norm L8C64-INT4 deployment residual exactly.

    This intentionally invokes the same Triton encoder and decoder as runtime
    rather than emulating Tensor Core dot products in PyTorch.
    """

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "runtime_bf16 RVQ assignment requires CUDA/Triton; use "
            "--rvq-assignment-mode fp32_proxy only for CPU diagnostics"
        )
    if codebook.dtype != torch.bfloat16:
        raise TypeError(
            "runtime_bf16 requires a BF16 INT4-dequantized codebook, got "
            f"{codebook.dtype}"
        )
    if codebook.ndim != 3 or codebook.shape[0] < 8:
        raise ValueError(
            f"Expected [L>=8,C,D] runtime codebook, got {tuple(codebook.shape)}"
        )
    from quant_videogen.kernels.triton.rvq import (
        rvq_decode_bhsd_triton,
        rvq_encode_bhsd_triton,
    )
    from quant_videogen.rvq import (
        _rvq_num_levels_limit,
        _rvq_runtime_codebook_cache_enabled,
        _rvq_triton_fastpath_enabled,
        _rvq_triton_block_tokens,
        _rvq_triton_decode_block_tokens,
    )

    if not _rvq_triton_fastpath_enabled():
        raise RuntimeError(
            "runtime_bf16 protocol explicitly requires the Triton fastpath, "
            "but HYWORLDPLAY_RVQ_TRITON_FASTPATH disables it"
        )
    if not _rvq_runtime_codebook_cache_enabled():
        raise RuntimeError(
            "runtime_bf16 Triton parity requires the resident runtime codebook "
            "cache, but HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE disables it"
        )
    if _rvq_num_levels_limit(int(codebook.shape[0])) != 8:
        raise RuntimeError(
            "runtime_bf16 strict-v2 requires exactly eight active RVQ levels; "
            "check HYWORLDPLAY_RVQ_NUM_LEVELS"
        )
    reference = x.to(device=device, dtype=torch.bfloat16).contiguous()
    bhsd = reference.reshape(1, 1, -1, reference.shape[-1]).contiguous()
    runtime_codebook = codebook[:8, None].contiguous()
    squared_norm = (
        runtime_codebook.float().square().sum(dim=-1).contiguous()
    )
    indices, _unused_scale = rvq_encode_bhsd_triton(
        bhsd,
        runtime_codebook,
        squared_norm,
        8,
        torch.uint8,
        block_tokens=_rvq_triton_block_tokens(),
        normalize=False,
    )
    decoded = rvq_decode_bhsd_triton(
        indices,
        runtime_codebook,
        None,
        torch.bfloat16,
        block_tokens=_rvq_triton_decode_block_tokens(),
        apply_scale=False,
    ).reshape_as(reference)
    residual = (reference - decoded).to(torch.bfloat16).contiguous()
    return decoded.contiguous(), residual


def rvq_residual(
    args: argparse.Namespace,
    x: torch.Tensor,
    codebook: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = assignment_mode(args)
    if mode == "runtime_bf16":
        return runtime_bf16_residual(x, codebook, device)
    return fp32_proxy_residual(x, codebook, device)


def deterministic_indices(
    total: int, count: int, generator: torch.Generator, mode: str
) -> torch.Tensor:
    count = min(total, count)
    if mode == "random":
        return torch.randperm(total, generator=generator)[:count]
    starts = list(range(0, total - count + 1, count))
    if not starts:
        return torch.arange(total)
    start = starts[int(torch.randint(len(starts), (1,), generator=generator))]
    return torch.arange(start, start + count)


def collect_training_residuals(
    args: argparse.Namespace, device: torch.device
) -> dict[str, torch.Tensor]:
    prompts = parse_ints(args.train_prompts)
    layers, heads = parse_ints(args.layers), parse_ints(args.heads)
    if len(prompts) != 20:
        raise ValueError("Formal training requires the fixed 20-prompt train split")
    per_unit = math.ceil(args.train_samples / (len(prompts) * args.train_units_per_prompt))
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in TENSOR_NAMES}
    for pi, prompt in enumerate(prompts):
        for unit in range(args.train_units_per_prompt):
            layer = layers[(pi * 7 + unit * 11) % len(layers)]
            head = heads[(pi * 13 + unit * 5) % len(heads)]
            saved = torch.load(
                str(layer_path(args.data_root, "train", prompt, layer)),
                map_location="cpu", weights_only=False, mmap=True,
            )
            for tensor_name in TENSOR_NAMES:
                x = saved[tensor_name][:, head].reshape(-1, DIM)
                gen = torch.Generator().manual_seed(
                    args.seed + prompt * 1_000_003 + layer * 1009 + head * 97
                    + (tensor_name == "V") * 17
                )
                ids = torch.randperm(x.shape[0], generator=gen)[:per_unit]
                _, residual = rvq_residual(
                    args, x[ids], load_int4_base_codebook(
                        args, tensor_name, layer, head, device
                    ), device
                )
                chunks[tensor_name].append(residual.cpu())
            print(
                f"train sample prompt={prompt} unit={unit + 1}/{args.train_units_per_prompt} "
                f"layer={layer} head={head}", flush=True
            )
    return {
        name: torch.cat(parts)[: args.train_samples].contiguous()
        for name, parts in chunks.items()
    }


def transform_for_training(
    x: torch.Tensor, group_size: int, hadamard: bool, seed: int
) -> torch.Tensor:
    if not hadamard:
        return x.float()
    module = codec_module()
    if not hasattr(module, "groupwise_hadamard"):
        raise AttributeError("quant_videogen.truncatable_residual.groupwise_hadamard is required")
    return module.groupwise_hadamard(x.float(), group_size, seed=seed, inverse=False)


def normalized_scalar_samples(
    residual: torch.Tensor, group_size: int, hadamard: bool, seed: int
) -> torch.Tensor:
    y = transform_for_training(residual, group_size, hadamard, seed)
    groups = y.reshape(-1, group_size)
    # Match encode_residual exactly: persist RMS as FP16, then normalize with
    # the dequantized FP16 value.  Fitting against an FP32-only scale optimizes
    # a subtly different scalar distribution.
    scale = (
        groups.square().mean(-1, keepdim=True).add(1e-12).sqrt()
        .to(torch.float16)
    )
    return (
        groups / scale.float().clamp_min(1e-8)
    ).reshape(-1).contiguous()


def binary_kmeans_1d(x: torch.Tensor, iterations: int) -> tuple[float, float, float]:
    if x.numel() == 0:
        return -1e-3, 1e-3, 0.0
    lo, hi = torch.quantile(x.float(), torch.tensor([0.25, 0.75], device=x.device))
    c0, c1 = lo, hi
    for _ in range(iterations):
        threshold = (c0 + c1) * 0.5
        right = x > threshold
        if (~right).any():
            c0 = x[~right].mean()
        if right.any():
            c1 = x[right].mean()
    if c0 > c1:
        c0, c1 = c1, c0
    return float(c0), float(c1), float((c0 + c1) * 0.5)


def _initial_gaussian_tree(
    x: torch.Tensor, iterations: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build the deterministic node-wise binary-k-means initialization."""

    values = x.float().cpu()
    parent = torch.zeros(values.numel(), dtype=torch.long)
    thresholds: list[torch.Tensor] = []
    centroids: list[torch.Tensor] = []
    previous = torch.zeros(1)
    for depth in range(1, 5):
        parents = 1 << (depth - 1)
        th = torch.empty(parents)
        child_centers = torch.empty(parents * 2)
        next_parent = torch.empty_like(parent)
        for node in range(parents):
            mask = parent == node
            if mask.any():
                c0, c1, threshold = binary_kmeans_1d(values[mask], iterations)
            else:
                center = float(previous[node]) if node < previous.numel() else 0.0
                c0, c1, threshold = center - 1e-4, center + 1e-4, center
            th[node] = threshold
            child_centers[2 * node] = c0
            child_centers[2 * node + 1] = c1
            if mask.any():
                next_parent[mask] = 2 * node + (values[mask] >= threshold).long()
        thresholds.append(th)
        centroids.append(child_centers)
        parent, previous = next_parent, child_centers
    return thresholds, centroids


def gaussian_tree_routes(
    values: torch.Tensor, thresholds: Iterable[torch.Tensor]
) -> list[torch.Tensor]:
    """Return the nested node id after every scalar-tree depth."""

    flat = values.float().reshape(-1)
    code = torch.zeros(flat.numel(), dtype=torch.long, device=flat.device)
    routes: list[torch.Tensor] = []
    for threshold in thresholds:
        local = threshold.to(device=flat.device, dtype=torch.float32)
        code = code * 2 + (flat >= local[code]).long()
        routes.append(code)
    return routes


def gaussian_tree_prefix_mse(
    values: torch.Tensor,
    thresholds: Iterable[torch.Tensor],
    centroids: Iterable[torch.Tensor],
) -> list[float]:
    flat = values.float().reshape(-1)
    routes = gaussian_tree_routes(flat, thresholds)
    return [
        float((flat - center.to(flat.device).float()[route]).square().mean())
        for route, center in zip(routes, centroids)
    ]


def gaussian_tree_objective(
    values: torch.Tensor,
    thresholds: Iterable[torch.Tensor],
    centroids: Iterable[torch.Tensor],
    weights: Iterable[float] = PREFIX_WEIGHTS,
) -> float:
    prefix = gaussian_tree_prefix_mse(values, thresholds, centroids)
    weight_values = tuple(float(value) for value in weights)
    if len(prefix) != len(weight_values):
        raise ValueError("Gaussian prefix/weight lengths disagree")
    return sum(weight * loss for weight, loss in zip(weight_values, prefix))


def _gaussian_forced_branch_cost(
    values: torch.Tensor,
    node: int,
    depth: int,
    branch: int,
    thresholds: list[torch.Tensor],
    centroids: list[torch.Tensor],
    weights: tuple[float, ...],
) -> torch.Tensor:
    """Cost at depths ``depth:`` after forcing one branch at ``depth``."""

    code = torch.full(
        (values.numel(),), 2 * node + branch, dtype=torch.long
    )
    cost = weights[depth] * (
        values - centroids[depth][code]
    ).square()
    for deeper in range(depth + 1, MAX_BITS):
        code = code * 2 + (values >= thresholds[deeper][code]).long()
        cost.add_(
            weights[deeper] * (values - centroids[deeper][code]).square()
        )
    return cost


def _recompute_gaussian_centroids(
    values: torch.Tensor,
    thresholds: list[torch.Tensor],
    centroids: list[torch.Tensor],
) -> list[torch.Tensor]:
    routes = gaussian_tree_routes(values, thresholds)
    updated: list[torch.Tensor] = []
    for depth, route in enumerate(routes):
        centers = centroids[depth].clone()
        count = 1 << (depth + 1)
        sums = torch.zeros(count, dtype=torch.float32)
        nums = torch.zeros(count, dtype=torch.float32)
        sums.index_add_(0, route, values)
        nums.index_add_(0, route, torch.ones_like(values))
        nonempty = nums > 0
        centers[nonempty] = sums[nonempty] / nums[nonempty]
        updated.append(centers)
    return updated


def fit_gaussian_tree(
    x: torch.Tensor,
    iterations: int,
    weights: Iterable[float] = PREFIX_WEIGHTS,
) -> dict[str, Any]:
    """Fit a nested tree against the auditable joint weighted-prefix loss.

    With centroids fixed, each threshold coordinate is solved over all exact
    boundaries between distinct samples.  With thresholds fixed, every prefix
    centroid is replaced by its assigned mean.  An update is committed only
    when the full joint objective cannot increase.
    """

    values = x.float().cpu().reshape(-1).contiguous()
    if values.numel() < 2:
        raise ValueError("Gaussian-tree fitting requires at least two values")
    weight_values = tuple(float(value) for value in weights)
    if len(weight_values) != MAX_BITS or any(value < 0 for value in weight_values):
        raise ValueError(f"Expected {MAX_BITS} non-negative prefix weights")
    thresholds, centroids = _initial_gaussian_tree(values, iterations)
    initial_packed_thresholds = [value.half() for value in thresholds]
    initial_packed_centroids = [value.half() for value in centroids]
    initial_packed_prefix = gaussian_tree_prefix_mse(
        values, initial_packed_thresholds, initial_packed_centroids
    )
    initial_packed_objective = sum(
        weight * loss
        for weight, loss in zip(weight_values, initial_packed_prefix)
    )
    current = gaussian_tree_objective(
        values, thresholds, centroids, weight_values
    )
    trace = [current]
    steps: list[dict[str, Any]] = []
    for sweep in range(max(1, int(iterations))):
        accepted = 0
        for depth in range(MAX_BITS):
            for node in range(1 << depth):
                routes = gaussian_tree_routes(values, thresholds[:depth])
                member = (
                    torch.ones(values.numel(), dtype=torch.bool)
                    if depth == 0
                    else routes[-1] == node
                )
                node_values = values[member]
                if node_values.numel() < 2:
                    continue
                left = _gaussian_forced_branch_cost(
                    node_values,
                    node,
                    depth,
                    0,
                    thresholds,
                    centroids,
                    weight_values,
                )
                right = _gaussian_forced_branch_cost(
                    node_values,
                    node,
                    depth,
                    1,
                    thresholds,
                    centroids,
                    weight_values,
                )
                order = torch.argsort(node_values)
                sorted_values = node_values[order]
                distinct = sorted_values[:-1] < sorted_values[1:]
                if not bool(distinct.any()):
                    continue
                left_prefix = left[order].double().cumsum(0)
                right_prefix = right[order].double().cumsum(0)
                split_cost = (
                    left_prefix[:-1] + right_prefix[-1] - right_prefix[:-1]
                )
                split_cost[~distinct] = float("inf")
                split = int(torch.argmin(split_cost)) + 1
                old_threshold = float(thresholds[depth][node])
                old_side = node_values >= old_threshold
                old_local = float(
                    torch.where(old_side, right, left).double().sum()
                )
                lower = sorted_values[split - 1]
                upper = sorted_values[split]
                candidate_threshold = float(torch.nextafter(lower, upper))
                candidate_side = node_values >= candidate_threshold
                candidate_local = float(
                    torch.where(candidate_side, right, left).double().sum()
                )
                tolerance = 1e-12 * max(1.0, abs(old_local))
                if candidate_local > old_local - tolerance:
                    continue
                before = current
                thresholds[depth][node] = candidate_threshold
                current += (candidate_local - old_local) / values.numel()
                if current > before + 1e-9 * max(1.0, abs(before)):
                    thresholds[depth][node] = old_threshold
                    current = before
                    continue
                accepted += 1
                trace.append(current)
                steps.append(
                    {
                        "kind": "threshold_coordinate",
                        "sweep": sweep,
                        "depth": depth + 1,
                        "node": node,
                        "objective_before": before,
                        "objective_after": current,
                    }
                )
        before_centroids = current
        candidate_centroids = _recompute_gaussian_centroids(
            values, thresholds, centroids
        )
        candidate_objective = gaussian_tree_objective(
            values, thresholds, candidate_centroids, weight_values
        )
        if candidate_objective <= before_centroids:
            centroids = candidate_centroids
            current = candidate_objective
            if candidate_objective < before_centroids:
                trace.append(current)
                steps.append(
                    {
                        "kind": "centroid_mean_update",
                        "sweep": sweep,
                        "objective_before": before_centroids,
                        "objective_after": current,
                    }
                )
        if accepted == 0 and current >= before_centroids:
            break

    packed_thresholds = [value.half() for value in thresholds]
    packed_centroids = [value.half() for value in centroids]
    packed_prefix = gaussian_tree_prefix_mse(
        values, packed_thresholds, packed_centroids
    )
    packed_objective = sum(
        weight * loss for weight, loss in zip(weight_values, packed_prefix)
    )
    packed_fallback_to_initial = packed_objective > initial_packed_objective
    if packed_fallback_to_initial:
        packed_thresholds = initial_packed_thresholds
        packed_centroids = initial_packed_centroids
        packed_prefix = initial_packed_prefix
        packed_objective = initial_packed_objective
    return {
        "thresholds": packed_thresholds,
        "centroids": packed_centroids,
        "fit_audit": {
            "algorithm": "exact_boundary_coordinate_descent_joint_prefix",
            "objective_definition": (
                "sum_b lambda_b * mean((u - centroid_b[path_prefix_b(u)])**2)"
            ),
            "prefix_weights": list(weight_values),
            "optimizer_objective_trace": trace,
            "optimizer_initial_objective": trace[0],
            "optimizer_final_objective": trace[-1],
            "initial_packed_fit_objective": initial_packed_objective,
            "packed_fit_prefix_MSE": packed_prefix,
            "packed_fit_objective": packed_objective,
            "packing_delta_vs_optimizer": packed_objective - trace[-1],
            "packed_fallback_to_initial": packed_fallback_to_initial,
            "packed_nonworse_than_initial": (
                packed_objective <= initial_packed_objective
            ),
            "accepted_steps": steps,
            "fit_values": int(values.numel()),
        },
    }


def additive_signs(
    values: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    error = values.float().reshape(-1).clone()
    signs: list[torch.Tensor] = []
    for step in alpha.float():
        sign = torch.where(error >= 0, 1.0, -1.0)
        signs.append(sign)
        error.sub_(step * sign)
    return torch.stack(signs, dim=1)


def additive_prefix_mse(
    values: torch.Tensor,
    alpha: torch.Tensor,
    signs: torch.Tensor | None = None,
) -> list[float]:
    flat = values.float().reshape(-1)
    if signs is None:
        signs = additive_signs(flat, alpha)
    reconstruction = torch.zeros_like(flat)
    result: list[float] = []
    for index, step in enumerate(alpha.float()):
        reconstruction.add_(step * signs[:, index])
        result.append(
            float((flat - reconstruction).double().square().mean())
        )
    return result


def additive_prefix_objective(
    values: torch.Tensor,
    alpha: torch.Tensor,
    weights: Iterable[float] = PREFIX_WEIGHTS,
    signs: torch.Tensor | None = None,
) -> float:
    prefix = additive_prefix_mse(values, alpha, signs)
    weight_values = tuple(float(value) for value in weights)
    if len(prefix) != len(weight_values):
        raise ValueError("Additive prefix/weight lengths disagree")
    return sum(weight * loss for weight, loss in zip(weight_values, prefix))


def fixed_sign_weighted_prefix_ls(
    values: torch.Tensor,
    signs: torch.Tensor,
    initial: torch.Tensor,
    weights: Iterable[float] = PREFIX_WEIGHTS,
    sweeps: int = 32,
) -> torch.Tensor:
    """Constrained coordinate solution of the all-prefix weighted LS problem."""

    flat = values.double().reshape(-1)
    design = signs.double()
    weight_values = tuple(float(value) for value in weights)
    if design.shape != (flat.numel(), MAX_BITS):
        raise ValueError(
            f"Expected fixed signs [{flat.numel()},{MAX_BITS}], got "
            f"{tuple(design.shape)}"
        )
    normal = torch.zeros(MAX_BITS, MAX_BITS, dtype=torch.float64)
    rhs = torch.zeros(MAX_BITS, dtype=torch.float64)
    for prefix, weight in enumerate(weight_values):
        if weight == 0:
            continue
        prefix_design = design[:, : prefix + 1]
        normal[: prefix + 1, : prefix + 1].add_(
            weight * (prefix_design.T @ prefix_design) / flat.numel()
        )
        rhs[: prefix + 1].add_(
            weight * (prefix_design.T @ flat) / flat.numel()
        )
    solution = initial.double().clone()
    margin = max(1e-6, float(solution[0]) * 1e-6)
    for _ in range(max(1, int(sweeps))):
        prior = solution.clone()
        for index in range(MAX_BITS):
            diagonal = float(normal[index, index])
            if diagonal <= 0:
                continue
            cross = torch.dot(normal[index], solution) - normal[index, index] * solution[index]
            optimum = (rhs[index] - cross) / normal[index, index]
            lower = (
                float(solution[index + 1]) + margin
                if index + 1 < MAX_BITS else margin
            )
            upper = (
                float(solution[index - 1]) - margin
                if index > 0 else float("inf")
            )
            if lower > upper:
                continue
            solution[index] = torch.clamp(
                optimum, min=lower, max=upper
            )
        if torch.equal(prior, solution):
            break
    if not bool((solution > 0).all()) or not bool(
        (solution[:-1] > solution[1:]).all()
    ):
        raise RuntimeError("Constrained additive LS lost strict alpha ordering")
    result = solution.float()
    if not bool((result > 0).all()) or not bool(
        (result[:-1] > result[1:]).all()
    ):
        return initial.float().clone()
    return result


def fit_additive_binary(
    x: torch.Tensor,
    iterations: int = 3,
    weights: Iterable[float] = PREFIX_WEIGHTS,
) -> dict[str, Any]:
    values = x.float().cpu().reshape(-1).contiguous()
    weight_values = tuple(float(value) for value in weights)
    if len(weight_values) != MAX_BITS or any(value < 0 for value in weight_values):
        raise ValueError(f"Expected {MAX_BITS} non-negative prefix weights")
    alpha: list[torch.Tensor] = []
    error = values.clone()
    for _ in range(MAX_BITS):
        step = error.abs().mean().clamp_min(1e-5)
        if alpha:
            step = torch.minimum(step, alpha[-1] * 0.99)
        alpha.append(step)
        error = error - step * torch.where(error >= 0, 1.0, -1.0)
    solution = torch.stack(alpha)
    initial_solution = solution.clone()
    current = additive_prefix_objective(
        values, solution, weight_values
    )
    trace = [current]
    fixed_sign_steps: list[dict[str, Any]] = []
    for iteration in range(max(1, int(iterations))):
        signs = additive_signs(values, solution)
        fixed_before = additive_prefix_objective(
            values, solution, weight_values, signs
        )
        ls_solution = fixed_sign_weighted_prefix_ls(
            values, signs, solution, weight_values
        )
        fixed_after = additive_prefix_objective(
            values, ls_solution, weight_values, signs
        )
        if fixed_after > fixed_before:
            # Casting the four-variable double solution back to FP32 can erase
            # a sub-ULP improvement.  Reject that candidate rather than
            # accepting or hiding a numerical increase.
            ls_solution = solution.clone()
            fixed_after = fixed_before
        accepted = False
        step_size = 1.0
        while step_size >= 1.0 / 1024.0:
            candidate = solution.lerp(ls_solution, step_size)
            candidate_objective = additive_prefix_objective(
                values, candidate, weight_values
            )
            if candidate_objective < current:
                solution = candidate
                current = candidate_objective
                trace.append(current)
                accepted = True
                break
            step_size *= 0.5
        fixed_sign_steps.append(
            {
                "iteration": iteration,
                "fixed_sign_objective_before": fixed_before,
                "fixed_sign_objective_after": fixed_after,
                "reassigned_objective_after": current,
                "accepted": accepted,
                "line_search_step": step_size if accepted else 0.0,
            }
        )
        if not accepted:
            break
    def pack_strict(candidate: torch.Tensor) -> torch.Tensor:
        packed_value = candidate.half().cpu()
        zero = torch.tensor(0, dtype=torch.float16)
        for i in range(1, MAX_BITS):
            if packed_value[i] >= packed_value[i - 1]:
                packed_value[i] = torch.nextafter(packed_value[i - 1], zero)
        if not bool((packed_value > 0).all()) or not bool(
            (packed_value[:-1] > packed_value[1:]).all()
        ):
            raise RuntimeError(
                "FP16 packing cannot represent four strictly decreasing "
                "positive additive coefficients"
            )
        return packed_value

    initial_packed = pack_strict(initial_solution)
    initial_packed_prefix = additive_prefix_mse(
        values, initial_packed.float()
    )
    initial_packed_objective = sum(
        weight * loss
        for weight, loss in zip(weight_values, initial_packed_prefix)
    )
    packed = pack_strict(solution)
    packed_prefix = additive_prefix_mse(values, packed.float())
    packed_objective = sum(
        weight * loss for weight, loss in zip(weight_values, packed_prefix)
    )
    packed_fallback_to_initial = packed_objective > initial_packed_objective
    if packed_fallback_to_initial:
        packed = initial_packed
        packed_prefix = initial_packed_prefix
        packed_objective = initial_packed_objective
    return {
        "alpha": packed,
        "fit_audit": {
            "algorithm": "alternating_fixed_sign_all_prefix_weighted_ls",
            "objective_definition": (
                "sum_b lambda_b * mean((u - "
                "sum_{i<=b} alpha_i*sign_i(u))**2)"
            ),
            "prefix_weights": list(weight_values),
            "optimizer_objective_trace": trace,
            "optimizer_initial_objective": trace[0],
            "optimizer_final_objective": trace[-1],
            "initial_packed_fit_objective": initial_packed_objective,
            "packed_fit_prefix_MSE": packed_prefix,
            "packed_fit_objective": packed_objective,
            "packing_delta_vs_optimizer": packed_objective - trace[-1],
            "packed_fallback_to_initial": packed_fallback_to_initial,
            "packed_nonworse_than_initial": (
                packed_objective <= initial_packed_objective
            ),
            "fixed_sign_steps": fixed_sign_steps,
            "fit_values": int(values.numel()),
        },
    }


def normalized_hvq_samples(
    residual: torch.Tensor,
    hadamard: bool,
    seed: int,
    limit: int,
    sample_seed: int,
) -> torch.Tensor:
    y = transform_for_training(residual, 64, hadamard, seed).reshape(-1, 64)
    scale = (
        y.square().mean(-1, keepdim=True).add(1e-12).sqrt()
        .to(torch.float16)
    )
    vectors = (
        y / scale.float().clamp_min(1e-8)
    ).reshape(-1, 4).contiguous()
    if vectors.shape[0] > limit:
        gen = torch.Generator().manual_seed(sample_seed)
        vectors = vectors[torch.randperm(vectors.shape[0], generator=gen)[:limit]]
    return vectors.float().cpu()


def segment_means(
    x: torch.Tensor, ids: torch.Tensor, count: int, fallback: torch.Tensor
) -> torch.Tensor:
    sums = torch.zeros(count, x.shape[1], dtype=x.dtype)
    nums = torch.zeros(count, dtype=x.dtype)
    sums.index_add_(0, ids, x)
    nums.index_add_(0, ids, torch.ones_like(ids, dtype=x.dtype))
    means = sums / nums.clamp_min(1).unsqueeze(1)
    empty = nums == 0
    means[empty] = fallback[empty]
    return means


def fit_hierarchical_vq(x: torch.Tensor, iterations: int) -> dict[str, list[torch.Tensor]]:
    """Fit a dense depth-16 binary 4-D VQ tree with vectorized node updates."""
    values = x.float().cpu()
    parent_ids = torch.zeros(values.shape[0], dtype=torch.long)
    parent_centers = values.mean(0, keepdim=True)
    tables: list[torch.Tensor] = []
    for depth in range(1, 17):
        parents = 1 << (depth - 1)
        sums = torch.zeros(parents, 4)
        sqs = torch.zeros(parents, 4)
        nums = torch.zeros(parents)
        sums.index_add_(0, parent_ids, values)
        sqs.index_add_(0, parent_ids, values.square())
        nums.index_add_(0, parent_ids, torch.ones(values.shape[0]))
        mean = sums / nums.clamp_min(1).unsqueeze(1)
        empty = nums == 0
        if parent_centers.shape[0] == parents:
            mean[empty] = parent_centers[empty]
        var = (sqs / nums.clamp_min(1).unsqueeze(1) - mean.square()).clamp_min(0)
        split_dim = var.argmax(1)
        delta = var.gather(1, split_dim[:, None]).sqrt().mul_(0.5).clamp_min_(1e-4)
        c0, c1 = mean.clone(), mean.clone()
        arange = torch.arange(parents)
        c0[arange, split_dim] -= delta[:, 0]
        c1[arange, split_dim] += delta[:, 0]
        for _ in range(iterations):
            d0 = (values - c0[parent_ids]).square().sum(1)
            d1 = (values - c1[parent_ids]).square().sum(1)
            side = (d1 < d0).long()
            child_ids = parent_ids * 2 + side
            fallback = torch.stack((c0, c1), 1).reshape(parents * 2, 4)
            children = segment_means(values, child_ids, parents * 2, fallback)
            c0, c1 = children[0::2], children[1::2]
        table = torch.stack((c0, c1), 1).reshape(parents * 2, 4)
        d0 = (values - table[(parent_ids * 2)]).square().sum(1)
        d1 = (values - table[(parent_ids * 2 + 1)]).square().sum(1)
        parent_ids = parent_ids * 2 + (d1 < d0).long()
        parent_centers = table
        tables.append(table.half())
        print(f"  HVQ depth={depth:02d} active={parent_ids.unique().numel():6d}/{1 << depth}", flush=True)
    return {"centroids": tables}


def fit_candidate_params(
    residuals: dict[str, torch.Tensor],
    scheme: str,
    hadamard: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tensor_name in TENSOR_NAMES:
        residual = residuals[tensor_name]
        if scheme == "nested_affine":
            result[tensor_name] = {}
            continue
        # Transform only the rows actually consumed by the trainer.  Building
        # all 400k x 128 normalized scalars would transiently allocate >200 MiB
        # per tensor/ablation and spend most FWHT work on discarded samples.
        if scheme == "hierarchical_vq":
            rows_needed = math.ceil(args.hvq_train_vectors / (DIM // 4))
        else:
            rows_needed = math.ceil(1_000_000 / DIM)
        rows_needed = min(int(residual.shape[0]), max(1, rows_needed))
        gen = torch.Generator().manual_seed(
            args.seed + 31 * (tensor_name == "V")
        )
        row_ids = torch.randperm(residual.shape[0], generator=gen)[:rows_needed]
        sampled_residual = residual[row_ids].contiguous()
        scalar = None
        if scheme != "hierarchical_vq":
            scalar = normalized_scalar_samples(
                sampled_residual, 64, hadamard, args.codec_seed
            )
        if scheme == "gaussian_tree":
            assert scalar is not None
            result[tensor_name] = fit_gaussian_tree(scalar, args.tree_iters)
        elif scheme == "additive_binary":
            assert scalar is not None
            result[tensor_name] = fit_additive_binary(
                scalar, args.additive_iters
            )
        elif scheme == "hierarchical_vq":
            vectors = normalized_hvq_samples(
                sampled_residual, hadamard, args.codec_seed, args.hvq_train_vectors,
                args.seed + 101 + (tensor_name == "V"),
            )
            result[tensor_name] = fit_hierarchical_vq(vectors, args.hvq_iters)
        else:
            raise ValueError(scheme)
    return result


def roundtrip_prefixes(
    residual: torch.Tensor,
    scheme: str,
    tensor_name: str,
    params: dict[str, Any],
    hadamard: bool,
    seed: int,
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    module = codec_module()
    full = module.encode_residual(
        residual, scheme, MAX_BITS, tensor_name,
        params_path=params, hadamard=hadamard, seed=seed,
    )
    decoded, sizes = {}, {}
    for bits in range(1, 5):
        prefix = module.truncate_state(full, bits)
        decoded[bits] = module.decode_residual(prefix, output_dtype=torch.float32)
        sizes[bits] = int(module.payload_nbytes(prefix))
    return decoded, sizes


def choose_transforms(rows: list[dict[str, Any]]) -> dict[str, bool]:
    selected: dict[str, bool] = {}
    for scheme in sorted({str(x["scheme"]) for x in rows}):
        scores: dict[bool, float] = {}
        for hadamard in (False, True):
            group = [x for x in rows if x["scheme"] == scheme and bool(x["hadamard"]) == hadamard]
            scores[hadamard] = sum(
                PREFIX_WEIGHTS[int(x["bits"]) - 1] * float(x["relative_residual_MSE"])
                for x in group
            ) / (2 * sum(PREFIX_WEIGHTS))
        selected[scheme] = min(scores, key=scores.get)
    return selected


def summarize_fit_objectives(
    candidates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Extract compact objective provenance without duplicating full traces."""

    summary: dict[str, Any] = {}
    for scheme, transforms in candidates.items():
        scheme_summary: dict[str, Any] = {}
        for label, tensors in transforms.items():
            tensor_summary: dict[str, Any] = {}
            for tensor_name, params in tensors.items():
                audit = params.get("fit_audit") if isinstance(params, dict) else None
                if not isinstance(audit, dict):
                    tensor_summary[tensor_name] = {
                        "objective": None,
                        "reason": "scheme_has_no_joint_scalar_fit_objective",
                    }
                    continue
                tensor_summary[tensor_name] = {
                    "algorithm": audit["algorithm"],
                    "objective_definition": audit["objective_definition"],
                    "prefix_weights": audit["prefix_weights"],
                    "optimizer_initial_objective": audit[
                        "optimizer_initial_objective"
                    ],
                    "optimizer_final_objective": audit[
                        "optimizer_final_objective"
                    ],
                    "packed_fit_objective": audit["packed_fit_objective"],
                    "packed_fit_prefix_MSE": audit["packed_fit_prefix_MSE"],
                    "fit_values": audit["fit_values"],
                    "accepted_optimizer_steps": max(
                        0, len(audit["optimizer_objective_trace"]) - 1
                    ),
                }
            scheme_summary[label] = tensor_summary
        summary[scheme] = scheme_summary
    return summary


def summarize_heldout_objectives(
    rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize the disjoint train-internal heldout selection objective."""

    result: dict[str, Any] = {}
    for scheme in sorted({str(row["scheme"]) for row in rows}):
        result[scheme] = {}
        for hadamard in (False, True):
            group = [
                row for row in rows
                if row["scheme"] == scheme
                and bool(row["hadamard"]) == hadamard
            ]
            by_tensor: dict[str, float] = {}
            for tensor_name in TENSOR_NAMES:
                tensor_rows = [
                    row for row in group if row["tensor"] == tensor_name
                ]
                by_tensor[tensor_name] = sum(
                    PREFIX_WEIGHTS[int(row["bits"]) - 1]
                    * float(row["relative_residual_MSE"])
                    for row in tensor_rows
                ) / sum(PREFIX_WEIGHTS)
            result[scheme]["hadamard" if hadamard else "direct"] = {
                "metric": (
                    "weighted relative residual MSE after physical codec "
                    "roundtrip on disjoint train-internal heldout vectors"
                ),
                "prefix_weights": list(PREFIX_WEIGHTS),
                "K": by_tensor["K"],
                "V": by_tensor["V"],
                "equal_weight_KV_mean": (
                    by_tensor["K"] + by_tensor["V"]
                ) / 2.0,
            }
    return result


def train(args: argparse.Namespace) -> None:
    training = args.output_root / "training"
    params_path = training / "codec_params.pt"
    manifest_path = training / "codec_params_manifest.json"
    metrics_path = training / "train_metrics.csv"
    selection_path = training / "selection.json"
    config_path = training / "config.json"
    run_info_path = training / "run_info.json"
    refuse_existing(
        params_path, manifest_path, metrics_path, selection_path,
        config_path, run_info_path,
    )
    device = device_from_args(args)
    residuals = collect_training_residuals(args, device)
    fit_residuals: dict[str, torch.Tensor] = {}
    eval_residuals: dict[str, torch.Tensor] = {}
    for tensor_index, tensor_name in enumerate(TENSOR_NAMES):
        values = residuals[tensor_name]
        if values.shape[0] < 2:
            raise ValueError("At least two residual vectors per tensor are required")
        eval_count = min(args.train_eval_samples, values.shape[0] - 1)
        generator = torch.Generator().manual_seed(
            args.seed + 9001 + tensor_index * 1009
        )
        order = torch.randperm(values.shape[0], generator=generator)
        eval_residuals[tensor_name] = values[order[:eval_count]].to(device)
        fit_residuals[tensor_name] = values[order[eval_count:]].contiguous()
    candidates: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for scheme in args.schemes:
        candidates[scheme] = {}
        for hadamard in (False, True):
            label = "hadamard" if hadamard else "direct"
            print(f"fit scheme={scheme} transform={label}", flush=True)
            fitted = fit_candidate_params(fit_residuals, scheme, hadamard, args)
            candidates[scheme][label] = fitted
            wrapper = {
                "schema_version": "truncatable_residual_params.v1",
                "schemes": {scheme: fitted},
            }
            for tensor_name in TENSOR_NAMES:
                residual = eval_residuals[tensor_name]
                decoded, sizes = roundtrip_prefixes(
                    residual, scheme, tensor_name, wrapper, hadamard, args.codec_seed
                )
                denom = float(residual.float().square().mean().clamp_min(1e-20))
                for bits, rec in decoded.items():
                    mse = float(F.mse_loss(rec.float(), residual.float()))
                    rows.append({
                        "scheme": scheme,
                        "hadamard": hadamard,
                        "tensor": tensor_name,
                        "bits": bits,
                        "residual_MSE": mse,
                        "relative_residual_MSE": mse / denom,
                        "payload_bytes": sizes[bits],
                        "payload_MiB": sizes[bits] / (1 << 20),
                        "compression_ratio_vs_BF16_residual": residual.numel() * 2 / sizes[bits],
                        "sample_role": "train_internal_heldout_selection",
                    })
            del wrapper
            if device.type == "cuda":
                torch.cuda.empty_cache()
    selected = choose_transforms(rows)
    fit_objectives = summarize_fit_objectives(candidates)
    heldout_objectives = summarize_heldout_objectives(rows)
    selected_params = {
        scheme: candidates[scheme]["hadamard" if selected[scheme] else "direct"]
        for scheme in args.schemes
    }
    payload = {
        "schema_version": "truncatable_residual_params.v1",
        "parameter_dtype": "float16",
        "schemes": selected_params,
        "ablation_candidates": candidates,
        "codec_seed": args.codec_seed,
        "training_samples_per_tensor": {
            k: int(v.shape[0]) for k, v in residuals.items()
        },
        "fit_samples_per_tensor": {
            k: int(v.shape[0]) for k, v in fit_residuals.items()
        },
        "heldout_selection_samples_per_tensor": {
            k: int(v.shape[0]) for k, v in eval_residuals.items()
        },
        "internal_split": (
            "deterministic random disjoint fit/selection split within the "
            "balanced 20-train-prompt pool"
        ),
        "prefix_loss_weights": list(PREFIX_WEIGHTS),
        "training_protocol_version": "truncatable_residual_strict_v2",
        "rvq_assignment": assignment_protocol_metadata(args),
        "fit_objective_summary": fit_objectives,
        "heldout_objective_summary": heldout_objectives,
    }
    selection = {
        "selected_hadamard": selected,
        "selection_basis": (
            "20-train-prompt-only internal held-out weighted mean K/V relative residual "
            "MSE across prefixes; fit and selection vectors are disjoint; "
            "lambda=[0.1,1.0,0.2,1.0]. No test prompt or VBench result was used."
        ),
        "parameter_dtype": "float16",
        "parameter_bytes_selected": tensor_nbytes(selected_params),
        "parameter_MiB_selected": tensor_nbytes(selected_params) / (1 << 20),
        "parameter_bytes_including_ablation_candidates": tensor_nbytes(candidates),
        "fit_objective_summary": fit_objectives,
        "heldout_objective_summary": heldout_objectives,
        "training_protocol_version": "truncatable_residual_strict_v2",
        "rvq_assignment": assignment_protocol_metadata(args),
    }
    write_csv(metrics_path, rows)
    atomic_torch_save(params_path, payload)
    atomic_json(manifest_path, {
        "schema_version": "truncatable_residual_params_manifest.v1",
        "codec_params_path": str(params_path),
        "codec_params_sha256": sha256_file(params_path),
        "parameter_dtype": "float16",
        "selected_parameter_bytes_by_scheme": {
            scheme: tensor_nbytes(selected_params[scheme])
            for scheme in args.schemes
        },
        "selected_parameter_MiB_by_scheme": {
            scheme: tensor_nbytes(selected_params[scheme]) / (1 << 20)
            for scheme in args.schemes
        },
        "selected_parameter_bytes_total": tensor_nbytes(selected_params),
        "ablation_candidate_parameter_bytes_total": tensor_nbytes(candidates),
        "accounting_note": (
            "Shared tables are excluded by payload_nbytes; offline/benchmark "
            "report them separately and charge them once in dataset-level ratios."
        ),
        "training_split_note": (
            "Codec parameters fit on a deterministic subset of the balanced "
            "20-train-prompt pool; Hadamard selection uses a disjoint "
            "train-internal held-out subset only."
        ),
        "training_protocol_version": "truncatable_residual_strict_v2",
        "rvq_assignment": assignment_protocol_metadata(args),
        "fit_objective_summary": fit_objectives,
        "heldout_objective_summary": heldout_objectives,
    })
    atomic_json(selection_path, selection)
    write_run_info(
        args, "train", config_path, run_info_path,
        parse_ints(args.train_prompts), "train",
        config_extra={
            "fit_objective_summary": fit_objectives,
            "heldout_objective_summary": heldout_objectives,
        },
        info_extra={
            "fit_objective_summary": fit_objectives,
            "heldout_objective_summary": heldout_objectives,
        },
    )
    print(json.dumps(selection, indent=2), flush=True)


def select(args: argparse.Namespace) -> None:
    training = args.output_root / "training"
    metrics_path, selection_path = training / "train_metrics.csv", training / "selection.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    normalized = [
        {**row, "bits": int(row["bits"]), "hadamard": row["hadamard"].lower() == "true"}
        for row in rows
    ]
    expected = choose_transforms(normalized)
    if selection_path.exists():
        saved = json.loads(selection_path.read_text(encoding="utf-8"))["selected_hadamard"]
        if saved != expected:
            raise RuntimeError(f"Stored selection differs from train-only recomputation: {saved} != {expected}")
        print(json.dumps({"validated": True, "selected_hadamard": saved}, indent=2))
        return
    atomic_json(selection_path, {
        "selected_hadamard": expected,
        "selection_basis": "recomputed from training/train_metrics.csv only",
    })


def load_training(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, bool]]:
    training = args.output_root / "training"
    params_path = training / "codec_params.pt"
    selection_path = training / "selection.json"
    manifest_path = training / "codec_params_manifest.json"
    config_path = training / "config.json"
    run_info_path = training / "run_info.json"
    required = (
        params_path,
        selection_path,
        manifest_path,
        config_path,
        run_info_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Strict-v2 training provenance is incomplete; missing=" + str(missing)
        )
    params_hash = sha256_file(params_path)
    selection_hash = sha256_file(selection_path)
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_config = json.loads(config_path.read_text(encoding="utf-8"))
    training_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    if manifest.get("codec_params_sha256") != params_hash:
        raise RuntimeError("Training manifest codec_params hash mismatch")
    expected_provenance = {
        "config_hash": sha256_file(config_path),
        "codec_params_hash": params_hash,
        "codec_params_manifest_hash": manifest_hash,
        "selection_hash": selection_hash,
    }
    mismatched = {
        key: {"stored": training_info.get(key), "actual": value}
        for key, value in expected_provenance.items()
        if training_info.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            "Training run_info provenance mismatch: "
            + json.dumps(mismatched, sort_keys=True)
        )
    if training_info.get("implementation_hash") != implementation_source_hashes():
        raise RuntimeError(
            "Training implementation hashes differ from the current strict-v2 "
            "source; do not mix code revisions"
        )
    current_codebook_hash = {
        "K": stable_directory_hash(
            args.data_root / "nonorm_k_l8c64_codebooks"
        ),
        "V": stable_directory_hash(
            args.data_root / "nonorm_v_l8c64_codebooks"
        ),
    }
    if training_info.get("codebook_hash") != current_codebook_hash:
        raise RuntimeError(
            "Training codebook directory hashes differ from the current "
            "L8C64 K/V codebooks"
        )
    params = torch.load(params_path, map_location="cpu", weights_only=False)
    protocol = params.get("training_protocol_version")
    if protocol != "truncatable_residual_strict_v2":
        raise RuntimeError(
            "Strict-v2 offline/benchmark refuses legacy codec parameters "
            f"(training_protocol_version={protocol!r}); retrain into a new "
            "--output-root"
        )
    stored_assignment = params.get("rvq_assignment", {}).get("mode")
    if stored_assignment != assignment_mode(args):
        raise RuntimeError(
            "Codec training/offline RVQ assignment modes disagree: "
            f"training={stored_assignment!r}, requested={assignment_mode(args)!r}"
        )
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    for label, payload_value in (
        ("params", params),
        ("selection", selection_payload),
        ("training config", training_config),
        ("training run_info", training_info),
    ):
        if payload_value.get("training_protocol_version") not in (
            None,
            "truncatable_residual_strict_v2",
        ):
            raise RuntimeError(f"{label} has a non-strict-v2 protocol")
        stored = payload_value.get("rvq_assignment", {}).get("mode")
        if stored != assignment_mode(args):
            raise RuntimeError(
                f"{label} assignment mode {stored!r} differs from "
                f"{assignment_mode(args)!r}"
            )
    selection = selection_payload["selected_hadamard"]
    return params, {k: bool(v) for k, v in selection.items()}


def validate_run_training_reference(
    args: argparse.Namespace, run_info_path: Path
) -> None:
    info = json.loads(run_info_path.read_text(encoding="utf-8"))
    training = args.output_root / "training"
    expected = {
        "codec_params_hash": sha256_file(training / "codec_params.pt"),
        "codec_params_manifest_hash": sha256_file(
            training / "codec_params_manifest.json"
        ),
        "selection_hash": sha256_file(training / "selection.json"),
    }
    mismatch = {
        key: {"stored": info.get(key), "actual": value}
        for key, value in expected.items()
        if info.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            "Run provenance references different training artifacts: "
            + json.dumps(mismatch, sort_keys=True)
        )


def attention_metrics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kh: torch.Tensor,
    vh: torch.Tensor,
    context: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float]:
    q, k, v, kh, vh = (x.float() for x in (q, k, v, kh, vh))
    if context is None:
        logits = q @ k.T / math.sqrt(DIM)
        probs = F.softmax(logits, -1)
        out = probs @ v
    else:
        logits, probs, out = context
    logits_h = q @ kh.T / math.sqrt(DIM)
    probs_h = F.softmax(logits_h, -1)
    out_h = probs_h @ vh
    k_mse = float(F.mse_loss(kh, k))
    v_mse = float(F.mse_loss(vh, v))
    return {
        "K_MSE": k_mse,
        "V_MSE": v_mse,
        "K_relative_MSE": k_mse / float(k.square().mean().clamp_min(1e-20)),
        "V_relative_MSE": v_mse / float(v.square().mean().clamp_min(1e-20)),
        "QK_logits_MSE": float(F.mse_loss(logits_h, logits)),
        "attention_probability_MSE": float(F.mse_loss(probs_h, probs)),
        "attention_KL": float(
            (probs * (probs.clamp_min(1e-8).log() - probs_h.clamp_min(1e-8).log()))
            .sum(-1).mean()
        ),
        "QKV_output_MSE": float(F.mse_loss(out_h, out)),
        "QKV_output_cosine": float(F.cosine_similarity(out_h, out, -1).mean()),
    }


def validate_formal_eval_axes(
    prompts: list[int],
    layers: list[int],
    heads: list[int],
    *,
    allow_partial: bool,
) -> None:
    """Validate the fixed formal test axes, allowing explicit diagnostic subsets."""

    actual = {
        "test prompts": tuple(prompts),
        "layers": tuple(layers),
        "heads": tuple(heads),
    }
    formal = {
        "test prompts": FORMAL_TEST_PROMPTS,
        "layers": FORMAL_LAYERS,
        "heads": FORMAL_HEADS,
    }
    for label, values in actual.items():
        if not values:
            raise ValueError(f"{label} must not be empty")
        outside = sorted(set(values) - set(formal[label]))
        if outside:
            raise ValueError(
                f"{label} contains values outside the fixed formal axis: {outside}"
            )
        if not allow_partial and values != formal[label]:
            raise ValueError(
                f"Formal offline evaluation requires {label}={list(formal[label])}; "
                "pass --allow-partial only for a diagnostic subset"
            )


def validate_formal_schemes(
    schemes: Iterable[str],
    *,
    allow_partial: bool,
) -> None:
    selected = tuple(schemes)
    if not selected:
        raise ValueError("At least one residual codec scheme is required")
    if not allow_partial and set(selected) != set(SCHEMES):
        raise ValueError(
            f"Formal offline evaluation requires all four schemes={list(SCHEMES)}; "
            "pass --allow-partial only for a diagnostic scheme subset"
        )


def expected_method_bits(schemes: Iterable[str]) -> dict[str, int]:
    result = {"l8c64_int4_base": 0}
    for scheme in schemes:
        for bits in range(1, 5):
            result[f"{scheme}_b{bits}"] = bits
    return result


def validate_offline_rows(
    rows: list[dict[str, Any]],
    expected_units: set[tuple[int, int, int]],
    schemes: Iterable[str],
    *,
    allow_partial: bool,
    expected_protocol: str = OFFLINE_RECONSTRUCTION_PROTOCOL,
    expected_assignment_mode: str | None = None,
) -> set[tuple[int, int, int]]:
    """Validate exact per-method unit coverage and return observed base units.

    ``allow_partial`` permits a strict subset of the requested units, but every
    requested method must still cover exactly the same subset.  This prevents a
    missing method (or an uneven prompt denominator) from silently entering a
    comparison.
    """

    if not rows:
        raise RuntimeError("No offline metric rows found")
    method_bits = expected_method_bits(schemes)
    units_by_method: dict[str, set[tuple[int, int, int]]] = {
        method: set() for method in method_bits
    }
    seen: set[tuple[str, int, int, int]] = set()
    unexpected_methods: set[str] = set()
    outside_units: set[tuple[int, int, int]] = set()
    for row in rows:
        method = str(row.get("method", ""))
        if method not in method_bits:
            unexpected_methods.add(method)
            continue
        try:
            bits = int(row["bits"])
            unit = (
                int(row["prompt_id"]),
                int(row["layer"]),
                int(row["head"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Malformed offline metric row: {row}") from error
        if bits != method_bits[method]:
            raise RuntimeError(
                f"Method {method!r} has bits={bits}, expected {method_bits[method]}"
            )
        if row.get("reconstruction_dtype") != expected_protocol:
            raise RuntimeError(
                f"Method {method!r} is missing reconstruction protocol "
                f"{expected_protocol!r}; refusing to mix runtime_bf16, legacy "
                "mislabelled, and explicitly named FP32-proxy rows"
            )
        if (
            expected_assignment_mode is not None
            and row.get("rvq_assignment_mode") != expected_assignment_mode
        ):
            raise RuntimeError(
                f"Method {method!r} has rvq_assignment_mode="
                f"{row.get('rvq_assignment_mode')!r}, expected "
                f"{expected_assignment_mode!r}"
            )
        for metric in OFFLINE_METRIC_NAMES:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Method {method!r} has a missing/non-numeric {metric}"
                ) from error
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Method {method!r} has non-finite {metric}={value}"
                )
        key = (method, *unit)
        if key in seen:
            raise RuntimeError(
                "Duplicate method/prompt/layer/head row found: "
                f"{method}/p{unit[0]}/l{unit[1]}/h{unit[2]}"
            )
        seen.add(key)
        units_by_method[method].add(unit)
        if unit not in expected_units:
            outside_units.add(unit)
    if unexpected_methods:
        raise RuntimeError(
            f"Unexpected or missing-name methods in offline rows: "
            f"{sorted(unexpected_methods)}"
        )
    if outside_units:
        raise RuntimeError(
            f"Offline rows contain units outside the requested axes: "
            f"{sorted(outside_units)[:10]}"
        )
    observed = units_by_method["l8c64_int4_base"]
    if not observed:
        raise RuntimeError("No l8c64_int4_base rows found")
    if not allow_partial and observed != expected_units:
        missing = sorted(expected_units - observed)
        raise RuntimeError(
            f"Incomplete base coverage: {len(observed)}/{len(expected_units)} units; "
            f"first missing units={missing[:10]}"
        )
    mismatched = {
        method: {
            "observed": len(units),
            "missing_vs_base": len(observed - units),
            "extra_vs_base": len(units - observed),
        }
        for method, units in units_by_method.items()
        if units != observed
    }
    if mismatched:
        raise RuntimeError(
            "Per-method unit coverage differs from the RVQ base: "
            + json.dumps(mismatched, sort_keys=True)
        )
    return observed


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise RuntimeError(f"{label} must be an explicit boolean, got {value!r}")


def _finite_number(
    row: dict[str, Any],
    key: str,
    *,
    positive: bool,
    allow_zero: bool = False,
) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Method {row.get('method')!r} has missing/non-numeric {key}"
        ) from error
    if not math.isfinite(value):
        raise RuntimeError(
            f"Method {row.get('method')!r} has non-finite {key}={value}"
        )
    if positive and value <= 0:
        raise RuntimeError(
            f"Method {row.get('method')!r} requires positive {key}, got {value}"
        )
    if allow_zero and value < 0:
        raise RuntimeError(
            f"Method {row.get('method')!r} requires non-negative {key}, got {value}"
        )
    return value


def validate_offline_payload_rows(
    rows: list[dict[str, Any]],
    schemes: Iterable[str],
    selected_hadamard: dict[str, bool] | None,
) -> None:
    """Validate method identity and all payload/rate fields before aggregation."""

    expected = expected_method_bits(schemes)
    observed_hadamard: dict[str, bool] = {}
    for row in rows:
        method = str(row["method"])
        bits = int(row["bits"])
        if method == "l8c64_int4_base":
            if row.get("scheme") != "rvq_base":
                raise RuntimeError("RVQ base row must use scheme='rvq_base'")
            payload = _finite_number(
                row, "residual_payload_bytes", positive=False, allow_zero=True
            )
            payload_mib = _finite_number(
                row, "residual_payload_MiB", positive=False, allow_zero=True
            )
            if payload != 0 or payload_mib != 0:
                raise RuntimeError("RVQ base residual payload must be exactly zero")
            ratio = row.get("residual_compression_ratio_vs_BF16")
            if ratio not in (None, "", "None"):
                raise RuntimeError("RVQ base residual compression ratio must be null")
            continue

        scheme = str(row.get("scheme", ""))
        if scheme not in schemes or method != f"{scheme}_b{bits}":
            raise RuntimeError(
                f"Method/scheme/bits mismatch: method={method!r}, "
                f"scheme={scheme!r}, bits={bits}"
            )
        if expected.get(method) != bits:
            raise RuntimeError(f"Unexpected method/rate row: {method}/b{bits}")
        hadamard = _strict_bool(
            row.get("hadamard"), f"{method}.hadamard"
        )
        prior = observed_hadamard.setdefault(scheme, hadamard)
        if prior != hadamard:
            raise RuntimeError(
                f"Scheme {scheme!r} mixes hadamard={prior} and {hadamard}"
            )
        if (
            selected_hadamard is not None
            and selected_hadamard.get(scheme) != hadamard
        ):
            raise RuntimeError(
                f"Scheme {scheme!r} row hadamard={hadamard} disagrees with "
                f"training selection={selected_hadamard.get(scheme)!r}"
            )

        key_tokens = _finite_number(
            row, "key_tokens", positive=True
        )
        query_tokens = _finite_number(
            row, "query_tokens", positive=True
        )
        if not key_tokens.is_integer() or not query_tokens.is_integer():
            raise RuntimeError("key_tokens/query_tokens must be positive integers")
        payload = _finite_number(
            row, "residual_payload_bytes", positive=True
        )
        if not payload.is_integer():
            raise RuntimeError("residual_payload_bytes must be an integer")
        payload_mib = _finite_number(
            row, "residual_payload_MiB", positive=True
        )
        ratio = _finite_number(
            row, "residual_compression_ratio_vs_BF16", positive=True
        )
        shared = _finite_number(
            row, "shared_parameter_bytes", positive=False, allow_zero=True
        )
        shared_mib = _finite_number(
            row, "shared_parameter_MiB", positive=False, allow_zero=True
        )
        combined_ratio = _finite_number(
            row, "single_unit_ratio_including_shared_parameters", positive=True
        )
        source_bytes = key_tokens * DIM * 4.0  # BF16 K and V residuals.
        expected_values = {
            "residual_payload_MiB": (payload_mib, payload / (1 << 20)),
            "residual_compression_ratio_vs_BF16": (
                ratio, source_bytes / payload
            ),
            "shared_parameter_MiB": (shared_mib, shared / (1 << 20)),
            "single_unit_ratio_including_shared_parameters": (
                combined_ratio, source_bytes / (payload + shared)
            ),
        }
        for label, (actual, expected_value) in expected_values.items():
            if not math.isclose(
                actual, expected_value, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise RuntimeError(
                    f"Method {method!r} has inconsistent {label}: "
                    f"{actual} != {expected_value}"
                )


def offline_unit(
    args: argparse.Namespace,
    prompt: int,
    layer: int,
    head: int,
    params: dict[str, Any],
    selected: dict[str, bool],
    device: torch.device,
) -> list[dict[str, Any]]:
    saved = torch.load(
        str(layer_path(args.data_root, "test", prompt, layer)),
        map_location="cpu", weights_only=False, mmap=True,
    )
    flat = {name: saved[name][:, head].reshape(-1, DIM) for name in ("Q", "K", "V")}
    gen = torch.Generator().manual_seed(args.seed + prompt * 1_000_003 + layer * 1009 + head)
    key_ids = deterministic_indices(flat["K"].shape[0], args.key_sample, gen, args.key_sampling)
    query_ids = torch.randperm(flat["Q"].shape[0], generator=gen)[: args.query_sample]
    q = flat["Q"][query_ids].to(device=device, dtype=torch.float32)
    k = flat["K"][key_ids].to(device=device, dtype=torch.float32)
    v = flat["V"][key_ids].to(device=device, dtype=torch.float32)
    kb, kr = rvq_residual(
        args, k, load_int4_base_codebook(args, "K", layer, head, device), device
    )
    vb, vr = rvq_residual(
        args, v, load_int4_base_codebook(args, "V", layer, head, device), device
    )
    reference_logits = q @ k.T / math.sqrt(DIM)
    reference_probs = F.softmax(reference_logits, -1)
    reference_context = (reference_logits, reference_probs, reference_probs @ v)
    common = {
        "prompt_id": prompt, "layer": layer, "head": head,
        "key_tokens": int(k.shape[0]), "query_tokens": int(q.shape[0]),
        "proxy_scope": "sampled_inference_span_without_positional_replay",
        "reconstruction_dtype": reconstruction_protocol(args),
        "rvq_assignment_mode": assignment_mode(args),
    }
    rows = [{
        **common, "method": "l8c64_int4_base", "scheme": "rvq_base", "bits": 0,
        "residual_payload_bytes": 0, "residual_payload_MiB": 0.0,
        "residual_compression_ratio_vs_BF16": None,
        **attention_metrics(q, k, v, kb, vb, reference_context),
    }]
    module = codec_module()
    params_source = args.output_root / "training" / "codec_params.pt"
    for scheme in args.schemes:
        shared_parameter_bytes = tensor_nbytes(params["schemes"].get(scheme, {}))
        full_k = module.encode_residual(
            kr, scheme, 4, "K", params_path=params_source,
            hadamard=selected[scheme], seed=args.codec_seed,
        )
        full_v = module.encode_residual(
            vr, scheme, 4, "V", params_path=params_source,
            hadamard=selected[scheme], seed=args.codec_seed,
        )
        for bits in range(1, 5):
            sk = module.truncate_state(full_k, bits)
            sv = module.truncate_state(full_v, bits)
            # Match uncompress_single_kv_cache exactly: both the RVQ base and
            # residual decode are BF16, and the BF16 sum is returned to the
            # attention kernel.  Keeping this addition in FP32 is measurably
            # optimistic at the highest residual prefix.
            k_res = module.decode_residual(sk, output_dtype=torch.bfloat16)
            v_res = module.decode_residual(sv, output_dtype=torch.bfloat16)
            kh = (kb + k_res).to(torch.bfloat16).contiguous()
            vh = (vb + v_res).to(torch.bfloat16).contiguous()
            payload_bytes = int(module.payload_nbytes(sk) + module.payload_nbytes(sv))
            source_bytes = (kr.numel() + vr.numel()) * 2
            rows.append({
                **common, "method": f"{scheme}_b{bits}", "scheme": scheme, "bits": bits,
                "hadamard": selected[scheme],
                "residual_payload_bytes": payload_bytes,
                "residual_payload_MiB": payload_bytes / (1 << 20),
                "shared_parameter_bytes": shared_parameter_bytes,
                "shared_parameter_MiB": shared_parameter_bytes / (1 << 20),
                "residual_compression_ratio_vs_BF16": source_bytes / payload_bytes,
                "single_unit_ratio_including_shared_parameters": (
                    source_bytes / (payload_bytes + shared_parameter_bytes)
                ),
                **attention_metrics(q, k, v, kh, vh, reference_context),
            })
    return rows


def offline(args: argparse.Namespace) -> None:
    rows_path = args.output_root / "offline" / f"shard_{args.shard_index:03d}.csv"
    config_path = args.output_root / "run_info" / f"offline_config_shard_{args.shard_index:03d}.json"
    run_info_path = args.output_root / "run_info" / f"offline_shard_{args.shard_index:03d}.json"
    refuse_existing(rows_path)
    prompts, layers, heads = (
        parse_ints(args.test_prompts), parse_ints(args.layers), parse_ints(args.heads)
    )
    validate_formal_eval_axes(
        prompts, layers, heads, allow_partial=bool(args.allow_partial)
    )
    validate_formal_schemes(
        args.schemes, allow_partial=bool(args.allow_partial)
    )
    if config_path.exists() or run_info_path.exists():
        if not (config_path.exists() and run_info_path.exists()):
            raise RuntimeError("Incomplete prior shard provenance: config/run_info must both exist")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != config_dict(args, "offline"):
            raise RuntimeError("Existing resumable shard config differs from current arguments")
    else:
        write_run_info(args, "offline", config_path, run_info_path, prompts, "test")
    params, selected = load_training(args)
    validate_run_training_reference(args, run_info_path)
    device = device_from_args(args)
    # Keep the ten prompts for a layer/head adjacent.  Combined with the
    # device-resident INT4 table cache, each codebook is loaded/quantized once.
    units = [
        (prompt, layer, head)
        for layer in layers for head in heads for prompt in prompts
    ]
    units = [unit for index, unit in enumerate(units) if index % args.num_shards == args.shard_index]
    rows: list[dict[str, Any]] = []
    unit_dir = args.output_root / "offline" / "units" / f"shard_{args.shard_index:03d}"
    started = time.time()
    for index, (prompt, layer, head) in enumerate(units):
        unit_path = unit_dir / f"p{prompt:02d}_l{layer:02d}_h{head:02d}.csv"
        if unit_path.exists():
            with unit_path.open(newline="", encoding="utf-8") as f:
                unit_rows = list(csv.DictReader(f))
        else:
            unit_rows = offline_unit(
                args, prompt, layer, head, params, selected, device
            )
        validate_offline_rows(
            unit_rows, {(prompt, layer, head)}, args.schemes,
            allow_partial=False,
            expected_protocol=reconstruction_protocol(args),
            expected_assignment_mode=assignment_mode(args),
        )
        if not unit_path.exists():
            write_csv(unit_path, unit_rows)
        rows.extend(unit_rows)
        if (index + 1) % 10 == 0 or index + 1 == len(units):
            elapsed = time.time() - started
            print(
                f"offline shard={args.shard_index} units={index + 1}/{len(units)} "
                f"elapsed={elapsed / 60:.1f}min", flush=True
            )
    write_csv(rows_path, rows)


def cuda_median_ms(
    fn: Callable[[], Any], warmup: int, trials: int
) -> tuple[float, Any]:
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    samples, result = [], None
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples), result


def benchmark_residuals(
    args: argparse.Namespace, device: torch.device
) -> dict[str, torch.Tensor]:
    prompt, layer = args.benchmark_prompt, args.benchmark_layer
    heads = parse_ints(args.heads)
    saved = torch.load(
        str(layer_path(args.data_root, "test", prompt, layer)),
        map_location="cpu", weights_only=False, mmap=True,
    )
    chunks = {name: [] for name in TENSOR_NAMES}
    for head in heads:
        gen = torch.Generator().manual_seed(args.seed + prompt * 1_000_003 + layer * 1009 + head)
        ids = deterministic_indices(
            saved["K"][:, head].numel() // DIM, args.key_sample, gen, args.key_sampling
        )
        for tensor_name in TENSOR_NAMES:
            x = saved[tensor_name][:, head].reshape(-1, DIM)[ids]
            _, residual = rvq_residual(
                args, x, load_int4_base_codebook(
                    args, tensor_name, layer, head, device
                ), device
            )
            chunks[tensor_name].append(residual)
    return {name: torch.stack(parts, 0).contiguous() for name, parts in chunks.items()}


def benchmark(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA events and a CUDA device")
    runtime_csv = args.output_root / "runtime" / "runtime.csv"
    runtime_json = args.output_root / "runtime" / "runtime.json"
    config_path = args.output_root / "run_info" / "benchmark_config.json"
    run_info_path = args.output_root / "run_info" / "benchmark.json"
    refuse_existing(runtime_csv, runtime_json, config_path, run_info_path)
    params, selected = load_training(args)
    device = device_from_args(args)
    residuals = benchmark_residuals(args, device)
    module = codec_module()
    params_source = args.output_root / "training" / "codec_params.pt"
    rows = []
    for scheme in args.schemes:
        shared_parameter_bytes = tensor_nbytes(params["schemes"].get(scheme, {}))
        def encode_pair() -> tuple[dict[str, Any], dict[str, Any]]:
            return tuple(
                module.encode_residual(
                    residuals[name], scheme, 4, name, params_path=params_source,
                    hadamard=selected[scheme], seed=args.codec_seed,
                ) for name in TENSOR_NAMES
            )

        encode_ms, states = cuda_median_ms(
            encode_pair, args.benchmark_warmup, args.benchmark_trials
        )
        for bits in range(1, 5):
            def truncate_pair() -> tuple[dict[str, Any], dict[str, Any]]:
                return tuple(module.truncate_state(state, bits) for state in states)

            truncate_ms, prefixes = cuda_median_ms(
                truncate_pair, args.benchmark_warmup, args.benchmark_trials
            )

            def decode_pair() -> tuple[torch.Tensor, torch.Tensor]:
                return tuple(
                    module.decode_residual(prefix, output_dtype=torch.bfloat16)
                    for prefix in prefixes
                )

            decode_ms, _ = cuda_median_ms(
                decode_pair, args.benchmark_warmup, args.benchmark_trials
            )
            payload_bytes = sum(int(module.payload_nbytes(x)) for x in prefixes)
            rows.append({
                "scheme": scheme, "bits": bits, "hadamard": selected[scheme],
                "shape_per_tensor": list(residuals["K"].shape),
                "elements_KV": residuals["K"].numel() + residuals["V"].numel(),
                "encode_B4_ms": encode_ms,
                "truncate_to_prefix_ms": truncate_ms,
                "decode_prefix_ms": decode_ms,
                "payload_bytes_KV": payload_bytes,
                "payload_MiB_KV": payload_bytes / (1 << 20),
                "shared_parameter_bytes": shared_parameter_bytes,
                "shared_parameter_MiB": shared_parameter_bytes / (1 << 20),
                "compression_ratio_vs_BF16_residual": (
                    (residuals["K"].numel() + residuals["V"].numel()) * 2 / payload_bytes
                ),
                "single_cache_ratio_including_shared_parameters": (
                    (residuals["K"].numel() + residuals["V"].numel()) * 2
                    / (payload_bytes + shared_parameter_bytes)
                ),
                "measurement_interval": "residual codec only; RVQ reconstruction excluded",
                "warmup": args.benchmark_warmup,
                "trials": args.benchmark_trials,
            })
        print(f"benchmark scheme={scheme} encode_B4={encode_ms:.3f}ms", flush=True)
    write_csv(runtime_csv, rows)
    atomic_json(runtime_json, rows)
    write_run_info(
        args, "benchmark", config_path, run_info_path,
        [args.benchmark_prompt], "test",
    )


def finite_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        raw = row.get(key)
        if raw in ("", None, "None"):
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def bootstrap_mean_ci(values: list[float], seed: int, samples: int) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap_mean_ci requires at least one value")
    if samples < 1:
        raise ValueError("bootstrap_mean_ci requires samples >= 1")
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    return means[round(0.025 * (samples - 1))], means[round(0.975 * (samples - 1))]


def validate_shard_provenance(
    args: argparse.Namespace,
    files: list[Path],
) -> list[dict[str, Any]]:
    """Validate compatible offline shard configs and return a hashed manifest."""

    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    configs: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    for csv_path in files:
        try:
            shard_index = int(csv_path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Malformed offline shard filename: {csv_path}") from error
        config_path = (
            args.output_root / "run_info"
            / f"offline_config_shard_{shard_index:03d}.json"
        )
        run_info_path = (
            args.output_root / "run_info"
            / f"offline_shard_{shard_index:03d}.json"
        )
        if not config_path.exists() or not run_info_path.exists():
            missing.extend(
                str(path) for path in (config_path, run_info_path)
                if not path.exists()
            )
            entries.append({
                "shard_index": shard_index,
                "csv_path": str(csv_path),
                "csv_sha256": sha256_file(csv_path),
                "provenance": "missing_diagnostic_only",
            })
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        info = json.loads(run_info_path.read_text(encoding="utf-8"))
        validate_run_training_reference(args, run_info_path)
        current_implementation = implementation_source_hashes()
        if info.get("implementation_hash") != current_implementation:
            raise RuntimeError(
                f"Offline shard {shard_index} implementation hash differs "
                "from the current strict-v2 source"
            )
        if config.get("implementation_source_hash") != current_implementation:
            raise RuntimeError(
                f"Offline shard {shard_index} config source hash differs "
                "from the current strict-v2 source"
            )
        if info.get("config_hash") != sha256_file(config_path):
            raise RuntimeError(
                f"Offline shard {shard_index} config hash does not match run_info"
            )
        if int(config.get("shard_index", -1)) != shard_index:
            raise RuntimeError(
                f"Offline shard filename/config index mismatch for {csv_path}"
            )
        configs.append(config)
        infos.append(info)
        entries.append({
            "shard_index": shard_index,
            "csv_path": str(csv_path),
            "csv_sha256": sha256_file(csv_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "run_info_path": str(run_info_path),
            "run_info_sha256": sha256_file(run_info_path),
            "provenance": "validated",
        })
    if missing:
        if not args.allow_partial:
            raise RuntimeError(
                "Formal aggregate requires config/run_info for every shard; "
                f"missing={missing}"
            )
        # A diagnostic aggregate may consume hand-built CSVs, but never accept
        # a mixture of provenance-bearing and provenance-free shards.
        if configs:
            raise RuntimeError(
                "Diagnostic aggregate cannot mix shards with and without provenance"
            )
        return entries

    declared_counts = {int(config.get("num_shards", -1)) for config in configs}
    if len(declared_counts) != 1:
        raise RuntimeError(
            f"Offline shard num_shards values disagree: {sorted(declared_counts)}"
        )
    declared_count = next(iter(declared_counts))
    indexes = {int(config["shard_index"]) for config in configs}
    if not args.allow_partial and indexes != set(range(declared_count)):
        raise RuntimeError(
            f"Offline shard files/configs are incomplete: observed={sorted(indexes)}, "
            f"expected={list(range(declared_count))}"
        )

    # Only fields that determine offline samples, codec states, and metrics are
    # compared to the aggregate invocation.  Shard index/count are validated
    # above; benchmark/training-only options do not affect these rows.
    current = config_dict(args, "offline")
    evaluation_fields = (
        "data_root",
        "output_root",
        "schemes",
        "test_prompts",
        "layers",
        "heads",
        "seed",
        "codec_seed",
        "device",
        "key_sample",
        "query_sample",
        "key_sampling",
        "rvq_assignment_mode",
        "allow_partial",
    )
    for shard_index, config in sorted(
        ((int(config["shard_index"]), config) for config in configs)
    ):
        differences = {
            key: {"shard": config.get(key), "aggregate": current.get(key)}
            for key in evaluation_fields
            if config.get(key) != current.get(key)
        }
        if differences:
            raise RuntimeError(
                f"Offline shard {shard_index} config is incompatible with aggregate: "
                + json.dumps(differences, sort_keys=True)
            )

    consistency_fields = (
        "git_commit",
        "codebook_hash",
        "codebook_marker_hash",
        "codec_params_hash",
        "codec_params_manifest_hash",
        "selection_hash",
        "implementation_hash",
        "rvq_assignment",
        "seed",
        "codec_seed",
        "num_shards",
    )
    for key in consistency_fields:
        serialized = {
            json.dumps(info.get(key), sort_keys=True) for info in infos
        }
        if len(serialized) != 1:
            raise RuntimeError(
                f"Offline shard run_info values disagree for {key}: {serialized}"
            )
    if not args.allow_partial and any(
        not info.get("implementation_hash") for info in infos
    ):
        raise RuntimeError(
            "Formal aggregate requires implementation hashes in every shard run_info"
        )
    return sorted(entries, key=lambda item: int(item["shard_index"]))


def aggregate(args: argparse.Namespace) -> None:
    out_dir = args.output_root / "aggregate"
    prompt_path, summary_csv, summary_json = (
        out_dir / "prompt_macro.csv", out_dir / "summary.csv", out_dir / "summary.json"
    )
    config_path = args.output_root / "run_info" / "aggregate_config.json"
    run_info_path = args.output_root / "run_info" / "aggregate.json"
    refuse_existing(prompt_path, summary_csv, summary_json, config_path, run_info_path)
    files = sorted((args.output_root / "offline").glob("shard_*.csv"))
    if not files:
        raise FileNotFoundError("No offline/shard_*.csv files")
    has_shard_provenance = any(
        (
            args.output_root / "run_info"
            / f"offline_shard_{int(path.stem.rsplit('_', 1)[1]):03d}.json"
        ).exists()
        for path in files
    )
    # Formal aggregation always revalidates the complete strict-v2 training
    # chain against current source/codebooks.  Provenance-bearing diagnostic
    # shards receive the same protection; only explicitly provenance-free
    # hand-built --allow-partial diagnostics bypass training artifacts.
    training_selected: dict[str, bool] | None = None
    if not args.allow_partial or has_shard_provenance:
        _params, training_selected = load_training(args)
    shard_manifest = validate_shard_provenance(args, files)
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    prompts, layers, heads = (
        parse_ints(args.test_prompts), parse_ints(args.layers), parse_ints(args.heads)
    )
    validate_formal_eval_axes(
        prompts, layers, heads, allow_partial=bool(args.allow_partial)
    )
    validate_formal_schemes(
        args.schemes, allow_partial=bool(args.allow_partial)
    )
    expected_unit_set = set(product(prompts, layers, heads))
    expected_units = len(expected_unit_set)
    observed_unit_set = validate_offline_rows(
        rows, expected_unit_set, args.schemes,
        allow_partial=bool(args.allow_partial),
        expected_protocol=reconstruction_protocol(args),
        expected_assignment_mode=assignment_mode(args),
    )
    validate_offline_payload_rows(
        rows, args.schemes, training_selected
    )
    observed_units = len(observed_unit_set)
    metric_names = (
        *OFFLINE_METRIC_NAMES,
        "residual_payload_MiB", "residual_compression_ratio_vs_BF16",
        "shared_parameter_MiB", "single_unit_ratio_including_shared_parameters",
    )
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["bits"]), int(row["prompt_id"]))].append(row)
    prompt_rows = []
    for (method, bits, prompt), group in sorted(grouped.items()):
        prompt_rows.append({
            "method": method, "bits": bits, "prompt_id": prompt,
            "units": len(group),
            **{metric: finite_mean(group, metric) for metric in metric_names},
        })
    summary_rows = []
    by_method: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        by_method[(row["method"], int(row["bits"]))].append(row)
    for (method, bits), group in sorted(by_method.items()):
        micro_group = [
            row for row in rows
            if row["method"] == method and int(row["bits"]) == bits
        ]
        record: dict[str, Any] = {
            "method": method, "bits": bits, "prompts": len(group),
            "coverage_units": observed_units, "expected_units": expected_units,
        }
        for metric in metric_names:
            vals = [float(x[metric]) for x in group if x.get(metric) not in (None, "", "None")]
            record[metric] = sum(vals) / len(vals) if vals else None
            if vals:
                low, high = bootstrap_mean_ci(
                    vals,
                    args.seed + int(hashlib.sha256(f"{method}:{bits}:{metric}".encode()).hexdigest()[:8], 16),
                    args.bootstrap_samples,
                )
                record[f"{metric}_bootstrap_ci95_low"] = low
                record[f"{metric}_bootstrap_ci95_high"] = high
        if method != "l8c64_int4_base":
            payload_sum = sum(float(x["residual_payload_bytes"]) for x in micro_group)
            source_sum = sum(
                float(x["residual_payload_bytes"])
                * float(x["residual_compression_ratio_vs_BF16"])
                for x in micro_group
            )
            shared_once = max(float(x["shared_parameter_bytes"]) for x in micro_group)
            record["evaluated_payload_MiB"] = payload_sum / (1 << 20)
            record["shared_parameter_MiB_charged_once"] = shared_once / (1 << 20)
            record["dataset_ratio_including_shared_parameters_once"] = (
                source_sum / (payload_sum + shared_once)
            )
        if method != "l8c64_int4_base":
            base = {
                int(x["prompt_id"]): float(x["QKV_output_MSE"])
                for x in prompt_rows if x["method"] == "l8c64_int4_base"
            }
            candidate = {int(x["prompt_id"]): float(x["QKV_output_MSE"]) for x in group}
            common = sorted(set(base) & set(candidate))
            improvements = [
                100.0 * (base[p] - candidate[p]) / max(base[p], 1e-30) for p in common
            ]
            base_mean = sum(base[p] for p in common) / len(common)
            candidate_mean = sum(candidate[p] for p in common) / len(common)
            record["prompt_wins_vs_RVQ_base"] = sum(candidate[p] < base[p] for p in common)
            record["QKV_output_MSE_mean_prompt_relative_improvement_pct"] = sum(improvements) / len(improvements)
            record["QKV_output_MSE_improvement_of_prompt_macro_means_pct"] = (
                100.0 * (base_mean - candidate_mean) / max(base_mean, 1e-30)
            )
            low, high = bootstrap_mean_ci(
                improvements,
                args.seed + int(hashlib.sha256(f"{method}:{bits}:improvement".encode()).hexdigest()[:8], 16),
                args.bootstrap_samples,
            )
            record["QKV_improvement_bootstrap_ci95_low"] = low
            record["QKV_improvement_bootstrap_ci95_high"] = high
        summary_rows.append(record)
    summary_payload = {
        "summary": summary_rows,
        "coverage": {
            "observed_units": observed_units,
            "expected_units": expected_units,
            "complete": observed_units == expected_units,
            "methods": len(expected_method_bits(args.schemes)),
            "per_method_coverage_identical": True,
        },
        "aggregation": (
            "arithmetic mean over available layer/head units within each prompt; "
            "equal-weight mean over prompt means; percentile bootstrap resamples "
            "prompt means (not individual layer/head rows)"
        ),
        "attention_scope": "offline proxy, not VBench and not exact online attention replay",
        "reconstruction_dtype": reconstruction_protocol(args),
        "rvq_assignment_mode": assignment_mode(args),
        "rate_accounting": {
            "baseline": "BF16 residual at 16 bits/value; L8C64 RVQ base excluded",
            "payload": (
                "physical packed bitplanes plus FP16 block metadata; "
                "container/header bytes excluded"
            ),
            "shared_parameters": (
                "selected K/V table bytes reported separately and charged once "
                "over all evaluated units"
            ),
        },
        "input_shards": shard_manifest,
    }
    # Complete every numerical/serialization check before creating any
    # aggregate artifact.  In particular, a non-finite derived JSON value can
    # no longer leave a valid-looking prompt_macro.csv behind.
    json.dumps(prompt_rows, allow_nan=False)
    json.dumps(summary_rows, allow_nan=False)
    json.dumps(summary_payload, allow_nan=False)
    write_csv(prompt_path, prompt_rows)
    write_csv(summary_csv, summary_rows)
    atomic_json(summary_json, summary_payload)
    write_run_info(
        args, "aggregate", config_path, run_info_path, prompts, "test",
        config_extra={"input_shards": shard_manifest},
        info_extra={"input_shards": shard_manifest},
    )
    print(json.dumps(summary_rows, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    {
        "train": train,
        "select": select,
        "offline": offline,
        "benchmark": benchmark,
        "aggregate": aggregate,
    }[args.command](args)


if __name__ == "__main__":
    main()
