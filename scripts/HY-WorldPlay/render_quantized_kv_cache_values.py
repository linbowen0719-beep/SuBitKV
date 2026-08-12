#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def parse_layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def find_span_dir(cache_dir: Path, span: str) -> Path:
    spans = sorted(path for path in cache_dir.glob("span_*_*") if path.is_dir())
    if not spans:
        raise FileNotFoundError(f"No span_*_* directories found under {cache_dir}")
    if span == "last":
        return spans[-1]
    if span == "first":
        return spans[0]
    span_dir = cache_dir / span
    if not span_dir.is_dir():
        raise FileNotFoundError(f"Span directory not found: {span_dir}")
    return span_dir


def span_tokens(span_dir: Path) -> tuple[int, int]:
    match = re.match(r"span_(\d+)_(\d+)$", span_dir.name)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def unpack_int4_residual(residual_quant: torch.Tensor, head: int, token_offset: int, num_tokens: int) -> np.ndarray:
    # residual_quant is packed as [B, H, S, D / 2]. Each byte is high nibble, low nibble.
    packed = residual_quant[0, head, token_offset : token_offset + num_tokens].to(torch.uint8)
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    values = torch.stack((high, low), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    return values.cpu().numpy().astype(np.uint8)


def reorder_tokens(
    values: np.ndarray,
    token_order: str,
    tokens_per_frame: int,
    frames_per_window: int,
) -> np.ndarray:
    if token_order == "frame_major":
        return values
    if token_order != "position_frame":
        raise ValueError(f"Unsupported token_order: {token_order}")
    expected_tokens = tokens_per_frame * frames_per_window
    if values.shape[0] != expected_tokens:
        raise ValueError(
            f"position_frame token order requires num_tokens={expected_tokens}, "
            f"got {values.shape[0]}"
        )

    channels = values.shape[1]
    return (
        values.reshape(frames_per_window, tokens_per_frame, channels)
        .transpose(1, 0, 2)
        .reshape(expected_tokens, channels)
    )


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def value_to_rgb(values: np.ndarray) -> np.ndarray:
    # Small viridis-like palette sampled for 0..15, kept local to avoid a matplotlib dependency.
    palette = np.array(
        [
            [68, 1, 84],
            [72, 26, 108],
            [71, 47, 125],
            [65, 68, 135],
            [57, 86, 140],
            [49, 104, 142],
            [42, 120, 142],
            [35, 136, 141],
            [31, 151, 139],
            [34, 167, 132],
            [53, 183, 121],
            [84, 197, 104],
            [122, 209, 81],
            [165, 219, 54],
            [210, 226, 27],
            [253, 231, 37],
        ],
        dtype=np.uint8,
    )
    return palette[np.clip(values, 0, 15)]


def render_grid(
    values: np.ndarray,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    cell_w: int,
    cell_h: int,
    font_size: int,
):
    num_rows, num_cols = values.shape
    left, top, right, bottom = 72, 46, 28, 42
    width = left + num_cols * cell_w + right
    height = top + num_rows * cell_h + bottom

    colors = value_to_rgb(values)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(16)
    tick_font = load_font(10)
    cell_font = load_font(font_size)

    draw.text((left, 12), title, fill=(20, 20, 20), font=title_font)
    draw.text((left + num_cols * cell_w // 2 - 35, height - 26), x_label, fill=(20, 20, 20), font=tick_font)
    draw.text((8, top + num_rows * cell_h // 2), y_label, fill=(20, 20, 20), font=tick_font)

    for row in range(num_rows):
        y = top + row * cell_h
        if row % 32 == 0:
            draw.text((left - 45, y), str(row), fill=(30, 30, 30), font=tick_font)
        for col in range(num_cols):
            x = left + col * cell_w
            rgb = tuple(int(v) for v in colors[row, col])
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), fill=rgb)
            value = int(values[row, col])
            luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            text_color = (0, 0, 0) if luminance > 145 else (235, 235, 235)
            draw.text((x + 1, y), str(value), fill=text_color, font=cell_font)

    for col in range(0, num_cols + 1, 16):
        x = left + col * cell_w
        draw.line((x, top, x, top + num_rows * cell_h), fill=(25, 25, 25), width=1)
        if col < num_cols:
            draw.text((x + 2, top - 17), str(col), fill=(30, 30, 30), font=tick_font)
    for row in range(0, num_rows + 1, 32):
        y = top + row * cell_h
        draw.line((left, y, left + num_cols * cell_w, y), fill=(25, 25, 25), width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--span", default="last", help="'first', 'last', or span_XXXXXX_YYYYYY")
    parser.add_argument("--layers", default="0,15,29")
    parser.add_argument("--kv", default="both", choices=["k", "v", "both"])
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--token_offset", type=int, default=0)
    parser.add_argument("--num_tokens", type=int, default=256)
    parser.add_argument(
        "--token_order",
        default="frame_major",
        choices=["frame_major", "position_frame"],
        help=(
            "frame_major keeps tokens as frame0 positions, frame1 positions, ...; "
            "position_frame reorders to position0 frames, position1 frames, ..."
        ),
    )
    parser.add_argument("--tokens_per_frame", type=int, default=880)
    parser.add_argument("--frames_per_window", type=int, default=4)
    parser.add_argument("--cell_w", type=int, default=10)
    parser.add_argument("--cell_h", type=int, default=10)
    parser.add_argument("--font_size", type=int, default=7)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    span_dir = find_span_dir(cache_dir, args.span)
    token_start, token_end = span_tokens(span_dir)
    output_dir = Path(args.output_dir)
    layers = parse_layers(args.layers)
    kv_names = ["k", "v"] if args.kv == "both" else [args.kv]

    summary = {
        "cache_dir": str(cache_dir),
        "span_dir": str(span_dir),
        "token_start": token_start,
        "token_end": token_end,
        "head": args.head,
        "token_offset": args.token_offset,
        "num_tokens": args.num_tokens,
        "token_order": args.token_order,
        "tokens_per_frame": args.tokens_per_frame,
        "frames_per_window": args.frames_per_window,
        "outputs": [],
    }

    for layer in layers:
        payload_path = span_dir / f"layer_{layer:02d}.pt"
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        for kv_name in kv_names:
            state = payload[kv_name]
            values = unpack_int4_residual(
                state["residual_quant"],
                head=args.head,
                token_offset=args.token_offset,
                num_tokens=args.num_tokens,
            )
            values = reorder_tokens(
                values,
                token_order=args.token_order,
                tokens_per_frame=args.tokens_per_frame,
                frames_per_window=args.frames_per_window,
            )
            name_suffix = (
                f"tokens{args.token_offset}_{args.token_offset + values.shape[0]}_"
                f"{args.token_order}_int4_values.png"
            )
            out_path = output_dir / (
                f"{span_dir.name}_layer{layer:02d}_{kv_name}_head{args.head:02d}_"
                f"{name_suffix}"
            )
            order_note = ""
            y_label = "token"
            if args.token_order == "position_frame":
                order_note = f" | pos-major {args.frames_per_window}f/pos"
                y_label = "position-major token"
            title = (
                f"{kv_name.upper()} INT4 residual | L{layer} H{args.head} | "
                f"tokens {token_start + args.token_offset}-{token_start + args.token_offset + values.shape[0]}"
                f"{order_note}"
            )
            render_grid(
                values,
                out_path,
                title=title,
                x_label="unpacked channel",
                y_label=y_label,
                cell_w=args.cell_w,
                cell_h=args.cell_h,
                font_size=args.font_size,
            )
            summary["outputs"].append(
                {
                    "path": str(out_path),
                    "layer": layer,
                    "kv": kv_name,
                    "shape": list(values.shape),
                    "min": int(values.min()),
                    "max": int(values.max()),
                    "mean": float(values.mean()),
                }
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
