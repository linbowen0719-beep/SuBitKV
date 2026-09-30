"""Merge ordered Wan KV spans into calibration samples."""

from __future__ import annotations

import shutil
from pathlib import Path


def layer_file(output_dir: Path, split_name: str, prompt_id: int, layer: int) -> Path:
    return output_dir / "kv_dump" / split_name / str(prompt_id) / f"layer_{layer}.pt"


def layer_file_done_for_tensor(
    output_dir: Path,
    split_name: str,
    prompt_id: int,
    layer: int,
    tensor_name: str,
    overwrite: bool,
) -> bool:
    if overwrite:
        return False
    path = layer_file(output_dir, split_name, prompt_id, layer)
    if not path.exists():
        return False
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return tensor_name in payload


def span_bounds(path: Path) -> tuple[int, int]:
    parts = path.stem.split("_")
    if len(parts) < 3 or parts[0] != "span":
        raise ValueError(f"Unexpected RVQ span filename: {path}")
    return int(parts[-2]), int(parts[-1])


def merge_prompt_spans(
    output_dir: Path,
    split_name: str,
    prompt_id: int,
    prompt: str,
    source_index: int,
    layers: list[int],
    n_spatial: int,
    overwrite: bool,
    tensor_name: str,
) -> None:
    import torch

    prompt_dir = output_dir / "kv_dump" / split_name / str(prompt_id)
    for layer in layers:
        out_path = prompt_dir / f"layer_{layer}.pt"
        span_dir = prompt_dir / "_spans" / f"layer_{layer}"
        if layer_file_done_for_tensor(
            output_dir,
            split_name,
            prompt_id,
            layer,
            tensor_name,
            overwrite,
        ):
            shutil.rmtree(span_dir, ignore_errors=True)
            continue
        span_paths = sorted(
            span_dir.glob("span_*.pt"), key=lambda path: span_bounds(path)[0]
        )
        if not span_paths:
            raise FileNotFoundError(f"No RVQ diagnostic spans found in {span_dir}")

        q_parts = []
        k_parts = []
        v_parts = []
        expected_start = 0
        for span_path in span_paths:
            start, end = span_bounds(span_path)
            if start != expected_start:
                raise ValueError(
                    f"Non-contiguous RVQ spans in {span_dir}: "
                    f"expected start {expected_start}, got {start} in {span_path.name}"
                )
            payload = torch.load(span_path, map_location="cpu", weights_only=False)
            q_parts.append(payload["Q"])
            k_parts.append(payload["K"])
            if "V" in payload:
                v_parts.append(payload["V"])
            expected_start = end
        q = torch.cat(q_parts, dim=2)
        k = torch.cat(k_parts, dim=2)
        v = torch.cat(v_parts, dim=2) if v_parts else None
        if q.shape != k.shape:
            raise ValueError(
                f"Q/K shape mismatch for {span_dir}: {q.shape} vs {k.shape}"
            )
        if v is not None and v.shape != k.shape:
            raise ValueError(
                f"V/K shape mismatch for {span_dir}: {v.shape} vs {k.shape}"
            )
        if q.shape[2] % n_spatial != 0:
            raise ValueError(
                f"Sequence length {q.shape[2]} is not divisible by n_spatial={n_spatial}"
            )
        t = q.shape[2] // n_spatial
        q = (
            q[0]
            .reshape(q.shape[1], t, n_spatial, q.shape[3])
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        k = (
            k[0]
            .reshape(k.shape[1], t, n_spatial, k.shape[3])
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        if v is not None:
            v = (
                v[0]
                .reshape(v.shape[1], t, n_spatial, v.shape[3])
                .permute(1, 0, 2, 3)
                .contiguous()
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "K": k,
            "Q": q,
            "metadata": {
                "prompt": prompt,
                "source_index": int(source_index),
                "split": split_name,
                "prompt_id": int(prompt_id),
                "layer": int(layer),
                "T": int(t),
                "H": int(k.shape[1]),
                "N": int(n_spatial),
                "D": int(k.shape[-1]),
                "key_type": "pre_rope",
                "query_type": "pre_rope",
                "value_type": "pre_prope" if v is not None else None,
                "layout": "THND",
                "note": "Pre-RoPE Q/K/V after projection/norm, before RoPE/PRoPE.",
            },
        }
        if v is not None:
            payload["V"] = v
        torch.save(
            payload,
            out_path,
        )
        shutil.rmtree(span_dir, ignore_errors=True)
        print(f"[dump] merged {out_path} shape={tuple(k.shape)}")
