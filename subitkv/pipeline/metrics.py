"""Video metrics with explicit frame pairing and reproducible TWE accounting."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .common import write_json


def warp(source, backward_flow):
    import torch
    import torch.nn.functional as F

    b, _, h, w = source.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=source.device),
        torch.arange(w, device=source.device),
        indexing="ij",
    )
    sample = torch.stack((xx, yy), -1).float()[None] + backward_flow.permute(0, 2, 3, 1)
    valid = (
        (sample[..., 0] >= 0)
        & (sample[..., 0] <= w - 1)
        & (sample[..., 1] >= 0)
        & (sample[..., 1] <= h - 1)
    )
    grid = torch.stack(
        (
            2 * sample[..., 0] / max(w - 1, 1) - 1,
            2 * sample[..., 1] / max(h - 1, 1) - 1,
        ),
        -1,
    )
    return F.grid_sample(
        source, grid, mode="bilinear", padding_mode="border", align_corners=True
    ), valid[:, None]


def twe_pairs(previous, current, forward, backward, config):
    """Per-pair luminance MAE in [0,1]; empty masks are invalid, never zero."""
    import torch

    warped, valid = warp(previous, backward)
    aligned_forward, _ = warp(forward, backward)
    consistency = (backward + aligned_forward).square().sum(1, keepdim=True)
    scale = backward.square().sum(1, keepdim=True) + aligned_forward.square().sum(
        1, keepdim=True
    )
    mask = valid & (
        consistency
        <= config["consistency_relative"] * scale + config["consistency_absolute"]
    )
    luminance = previous.new_tensor(config["luminance"]).view(1, 3, 1, 1)
    error = (
        (current * luminance).sum(1, keepdim=True)
        - (warped * luminance).sum(1, keepdim=True)
    ).abs()
    count = mask.flatten(1).sum(1)
    values = (error * mask).flatten(1).sum(1) / count.clamp_min(1)
    values = torch.where(count > 0, values, torch.full_like(values, float("nan")))
    return values, mask.float().flatten(1).mean(1)


def read_video(path: Path):
    import av
    import numpy as np

    with av.open(str(path)) as video:
        frames = [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
    if len(frames) < 2:
        raise ValueError(f"video requires at least two frames: {path}")
    return np.stack(frames)


def temporal_error(frames, flow_model, config, *, device, boundaries, exclude=0):
    import cv2
    import numpy as np
    import torch

    resized = np.stack(
        [
            cv2.resize(
                frame, (config["width"], config["height"]), interpolation=cv2.INTER_AREA
            )
            for frame in frames
        ]
    )
    tensors = torch.from_numpy(resized).permute(0, 3, 1, 2).float() / 255
    rows = []
    batch = int(config["batch_size"])
    with torch.inference_mode():
        for start in range(exclude, len(tensors) - 1, batch):
            stop = min(start + batch, len(tensors) - 1)
            previous, current = (
                tensors[start:stop].to(device),
                tensors[start + 1 : stop + 1].to(device),
            )
            forward = flow_model(previous * 2 - 1, current * 2 - 1)[-1]
            backward = flow_model(current * 2 - 1, previous * 2 - 1)[-1]
            values, fractions = twe_pairs(previous, current, forward, backward, config)
            for offset, (value, fraction) in enumerate(
                zip(values.tolist(), fractions.tolist())
            ):
                frame = start + offset + 1
                rows.append(
                    dict(
                        frame_from=frame - 1,
                        frame_to=frame,
                        boundary=frame in boundaries,
                        twe=value if math.isfinite(value) else None,
                        valid_fraction=fraction,
                    )
                )
    return rows


def finite_mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(values) / len(values) if values else None


def summarize_twe(rows):
    return {
        "twe": finite_mean(r["twe"] for r in rows),
        "twe_interior": finite_mean(r["twe"] for r in rows if not r["boundary"]),
        "twe_boundary": finite_mean(r["twe"] for r in rows if r["boundary"]),
        "twe_valid_pairs": sum(r["twe"] is not None for r in rows),
        "twe_invalid_pairs": sum(r["twe"] is None for r in rows),
    }


def similarity(frames, reference, enabled, lpips_model, device):
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity

    if frames.shape != reference.shape:
        raise ValueError(
            f"reference shape mismatch: {frames.shape} != {reference.shape}"
        )
    per_frame = {name: [] for name in ("psnr", "ssim", "lpips") if name in enabled}
    identical = 0
    for frame, ref in zip(frames, reference):
        x, y = frame.astype(np.float32) / 255, ref.astype(np.float32) / 255
        if "psnr" in per_frame:
            mse = float(np.square(x.astype(np.float64) - y).mean())
            identical += mse == 0
            per_frame["psnr"].append(
                float("inf") if mse == 0 else -10 * math.log10(mse)
            )
        if "ssim" in per_frame:
            per_frame["ssim"].append(
                float(
                    structural_similarity(
                        x,
                        y,
                        data_range=1,
                        channel_axis=-1,
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                    )
                )
            )
        if "lpips" in per_frame:
            a = torch.from_numpy(x).permute(2, 0, 1)[None].to(device) * 2 - 1
            b = torch.from_numpy(y).permute(2, 0, 1)[None].to(device) * 2 - 1
            with torch.inference_mode():
                per_frame["lpips"].append(float(lpips_model(a, b).item()))
    result = {key: sum(values) / len(values) for key, values in per_frame.items()}
    # Preserve infinite PSNR without emitting invalid JSON or silently dropping frames.
    if "psnr" in result and math.isinf(result["psnr"]):
        result["psnr"] = "inf"
    result["identical_frames"] = identical
    return result


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config, records, output):
    import torch

    enabled, device = config["metrics"]["enabled"], config["runtime"]["device"]
    m = config["model_config"]
    reference = {r["prompt_id"]: r for r in records if r["method"] == "bf16"}
    flow_model = lpips_model = None
    provenance = {
        "config": config["metrics"],
        "aggregation": "frame mean per video, then equal video mean",
    }
    if "twe" in enabled:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        weights = Raft_Small_Weights[config["metrics"]["twe"]["weights"]]
        flow_model = raft_small(weights=weights, progress=True).eval().to(device)
        provenance["flow_weights_url"] = weights.url
    if "lpips" in enabled:
        import lpips

        lpips_model = (
            lpips.LPIPS(net=config["metrics"]["lpips_network"]).eval().to(device)
        )
    rows, pairs = [], []
    for record in records:
        row = {k: record[k] for k in ("method", "prompt_id", "seed")}
        try:
            frames = read_video(Path(record["video"]))
            if len(frames) != m["frames"]:
                raise ValueError(f"expected {m['frames']} frames, got {len(frames)}")
            if "twe" in enabled:
                temporal = temporal_error(
                    frames,
                    flow_model,
                    config["metrics"]["twe"],
                    device=device,
                    boundaries=record["chunk_boundaries"],
                    exclude=m["twe_exclude_frames"],
                )
                pairs.extend(dict(**row, **pair) for pair in temporal)
                row.update(summarize_twe(temporal))
            if any(name in enabled for name in ("psnr", "ssim", "lpips")):
                ref = reference[record["prompt_id"]]
                if (
                    ref["seed"] != record["seed"]
                    or ref["input_hash"] != record["input_hash"]
                ):
                    raise ValueError("BF16 reference seed/input mismatch")
                exclude = m["similarity_exclude_frames"]
                row.update(
                    similarity(
                        frames[exclude:],
                        read_video(Path(ref["video"]))[exclude:],
                        enabled,
                        lpips_model,
                        device,
                    )
                )
            row.update(record.get("runtime_metrics", {}))
            baseline = row.get("bf16_kv_payload_mib") or reference.get(
                record["prompt_id"], {}
            ).get("runtime_metrics", {}).get("kv_payload_mib")
            payload = row.get("kv_payload_mib")
            row["compression_ratio_vs_bf16"] = (
                baseline / payload if baseline and payload else None
            )
            row["status"] = "success"
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "per_video.csv", rows)
    if pairs:
        csv_write(output / "twe_per_pair.csv", pairs)
    summaries = []
    for method in dict.fromkeys(r["method"] for r in rows):
        selected = [r for r in rows if r["method"] == method]
        good = [r for r in selected if r["status"] == "success"]
        result = dict(
            method=method,
            expected=len(selected),
            valid=len(good),
            failed=len(selected) - len(good),
        )
        for key in (
            "psnr",
            "ssim",
            "lpips",
            "twe",
            "twe_interior",
            "twe_boundary",
            "k_payload_mib",
            "v_payload_mib",
            "kv_payload_mib",
            "kv_peak_payload_mib",
            "shared_codebook_mib",
            "compression_ratio_vs_bf16",
            "generation_seconds",
            "process_seconds",
        ):
            result[key] = (
                "inf"
                if any(r.get(key) == "inf" for r in good)
                else finite_mean(r.get(key) for r in good)
            )
        summaries.append(result)
    csv_write(output / "summary.csv", summaries)
    write_json(output / "summary.json", dict(results=summaries, provenance=provenance))
    if any(r["status"] != "success" for r in rows):
        raise RuntimeError(
            f"Some videos failed evaluation; see {output / 'per_video.csv'}"
        )
    if "vbench" in enabled:
        from vbench import VBench

        # One isolated directory per method, with VBench custom-input naming.
        for method in dict.fromkeys(r["method"] for r in records):
            directory = output / "vbench_inputs" / method
            directory.mkdir(parents=True, exist_ok=False)
            prompt_map = {}
            for r in records:
                if r["method"] != method:
                    continue
                target = directory / f"{r['prompt_id']:04d}.mp4"
                target.symlink_to(Path(r["video"]).resolve())
                prompt_map[target.name] = r["prompt"]
            evaluator = VBench(torch.device(device), None, str(output / "vbench"))
            evaluator.evaluate(
                videos_path=str(directory),
                name=method,
                prompt_list=prompt_map,
                dimension_list=config["metrics"]["vbench_dimensions"],
                mode="custom_input",
            )
