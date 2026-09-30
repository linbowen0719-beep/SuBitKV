#!/usr/bin/env python3
"""Offline and video-command bench for L8C64 INT4 RVQ error feedback.

The offline stage replays saved pre-RoPE/pre-PRoPE K/V dumps chunk by chunk,
keeps the existing L8C64 codebooks fixed, and changes only RVQ code
assignment. Packed RVQ states remain decodable by the normal runtime decoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
LAYERS = 30
HEADS = 24
CHUNK_FRAMES = 4
TOKENS_PER_FRAME = 880
RVQ_STORAGE = "int4"


@dataclass(frozen=True)
class Setting:
    name: str
    method: str
    scope: str
    lam: float = 0.0
    beta: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--prompt-ids", default="0,1,2")
    parser.add_argument("--layers", default=",".join(str(i) for i in range(LAYERS)))
    parser.add_argument("--methods", default="inner,input_comp,attn_output")
    parser.add_argument("--scopes", default="k,v,kv")
    parser.add_argument("--inner-lambdas", default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--input-betas", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--attn-lambdas", default="0.01,0.05,0.1,0.25")
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument("--key-sample", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-settings", type=int, default=0, help="Debug limit; 0 runs the full grid.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--write-video-commands", action="store_true", default=True)
    return parser.parse_args()


def int_list(raw: str) -> list[int]:
    if raw.strip().lower() == "all":
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def str_list(raw: str) -> list[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def clean_name(value: str) -> str:
    return value.lower().replace("+", "plus").replace("/", "_").replace("-", "_").replace(" ", "_")


def make_settings(args: argparse.Namespace) -> list[Setting]:
    settings = [Setting("baseline", "none", "kv")]
    methods = set(str_list(args.methods))
    scopes = str_list(args.scopes)
    if "inner" in methods:
        for lam in float_list(args.inner_lambdas):
            for scope in scopes:
                settings.append(Setting(f"inner_l{lam:g}_{scope}", "inner", scope, lam=lam))
    if "input_comp" in methods:
        for beta in float_list(args.input_betas):
            for scope in scopes:
                settings.append(Setting(f"input_b{beta:g}_{scope}", "input_comp", scope, beta=beta))
    if "attn_output" in methods:
        for lam in float_list(args.attn_lambdas):
            for scope in scopes:
                settings.append(Setting(f"attn_l{lam:g}_{scope}", "attn_output", scope, lam=lam))
    if args.max_settings > 0:
        settings = settings[: args.max_settings]
    return settings


def codebook_dir(root: Path, tensor_name: str) -> Path:
    return root / f"sweep_{tensor_name.lower()}_l8c64_codebooks"


def layer_path(root: Path, split: str, prompt_id: int, layer: int) -> Path:
    return root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"


def thnd_to_bhsd(x: torch.Tensor) -> torch.Tensor:
    # [T,H,N,D] -> [1,H,T*N,D]
    return x.permute(1, 0, 2, 3).reshape(x.shape[1], x.shape[0] * x.shape[2], x.shape[3])[None]


def bhsd_to_thnd(x: torch.Tensor, frames: int, tokens_per_frame: int = TOKENS_PER_FRAME) -> torch.Tensor:
    # [1,H,T*N,D] -> [T,H,N,D]
    return x[0].reshape(x.shape[1], frames, tokens_per_frame, x.shape[-1]).permute(1, 0, 2, 3).contiguous()


def scope_enabled(setting: Setting, tensor_name: str) -> bool:
    if setting.method == "none":
        return False
    if setting.scope == "kv":
        return True
    return setting.scope == tensor_name.lower()


def codebook_storage(storage: str) -> tuple[str, str]:
    if storage == "int4":
        return "int4", "uint4"
    if storage == "int8":
        return "int8", "uint8"
    return "bf16", "bf16"


def sample_indices(total: int, take: int, seed: int) -> torch.Tensor:
    take = min(take, total)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(total, generator=gen)[:take]


def prepare_q_samples(q: torch.Tensor, query_sample: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    q_flat = q.permute(1, 0, 2, 3).reshape(q.shape[1], q.shape[0] * q.shape[2], q.shape[-1])
    q_idx = sample_indices(q_flat.shape[1], query_sample, seed)
    return q_flat.index_select(1, q_idx).to(device=device, dtype=torch.float32), q_idx


def v_guidance(
    q_sample: torch.Tensor,
    k_chunk: torch.Tensor,
    output_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # q_sample [H,Q,D], k_chunk [1,H,S,D], output_error [H,Q,D]
    k_hsd = k_chunk[0].float()
    logits = torch.bmm(q_sample, k_hsd.transpose(1, 2)) / math.sqrt(k_hsd.shape[-1])
    attn = torch.softmax(logits.float(), dim=-1)
    guidance = torch.einsum("hqs,hqd->hsd", attn, output_error)
    curvature = (attn * attn).sum(dim=1)[:, :, None]
    return guidance[None], curvature[None], attn


def k_guidance(
    q_sample: torch.Tensor,
    logit_error: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # logit_error [H,Q] approximates accumulated past logit error per sampled query.
    dim = q_sample.shape[-1]
    guidance = torch.einsum("hq,hqd->hd", logit_error, q_sample) / math.sqrt(dim)
    guidance = guidance[:, None, :].expand(-1, seq_len, -1).contiguous()
    curvature = (q_sample.float() * q_sample.float()).sum(dim=-1).sum(dim=1)[:, None, None] / float(dim)
    curvature = curvature.expand(-1, seq_len, 1).contiguous()
    return guidance[None], curvature[None]


@torch.no_grad()
def encode_chunk(
    chunk: torch.Tensor,
    *,
    root: Path,
    tensor_name: str,
    layer: int,
    setting: Setting,
    state: dict[str, Any],
    device: torch.device,
    batch_size: int,
    guidance: torch.Tensor | None = None,
    curvature: torch.Tensor | None = None,
) -> torch.Tensor:
    from subitkv.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache

    codebook_dtype, scale_dtype = codebook_storage(RVQ_STORAGE)
    active = scope_enabled(setting, tensor_name)
    method = setting.method if active else "none"
    packed = rvq_encode_tensor_cache(
        chunk.to(device=device),
        layer_idx=layer,
        codebook_dir=codebook_dir(root, tensor_name.upper()),
        batch_size=batch_size,
        compute_dtype="bf16",
        codebook_storage_dtype=codebook_dtype,
        scale_storage_dtype=scale_dtype,
        feedback_method=method,
        feedback_lambda=setting.lam,
        feedback_beta=setting.beta,
        feedback_state=state if active else None,
        feedback_guidance=guidance if active and method == "attn_output" else None,
        feedback_curvature=curvature if active and method == "attn_output" else None,
    )
    return rvq_dequantize_tensor_cache(packed, chunk.dtype).detach()


@torch.no_grad()
def reconstruct_prompt_layer(
    payload: dict[str, Any],
    *,
    root: Path,
    layer: int,
    setting: Setting,
    device: torch.device,
    batch_size: int,
    query_sample: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = payload["Q"].float()
    k = payload["K"].float()
    v = payload["V"].float()
    total_frames = k.shape[0]
    chunks = [(start, min(start + CHUNK_FRAMES, total_frames)) for start in range(0, total_frames, CHUNK_FRAMES)]
    q_sample, _ = prepare_q_samples(q, query_sample, seed, device)
    v_output_error = torch.zeros((HEADS, q_sample.shape[1], v.shape[-1]), device=device, dtype=torch.float32)
    k_logit_error = torch.zeros((HEADS, q_sample.shape[1]), device=device, dtype=torch.float32)
    states = {"k": {}, "v": {}}
    k_hat_chunks: list[torch.Tensor] = []
    v_hat_chunks: list[torch.Tensor] = []

    for start, end in chunks:
        k_chunk = thnd_to_bhsd(k[start:end])
        v_chunk = thnd_to_bhsd(v[start:end])

        k_guide = k_curv = None
        if setting.method == "attn_output" and scope_enabled(setting, "k"):
            k_guide, k_curv = k_guidance(q_sample, k_logit_error, k_chunk.shape[2])
        k_hat_bhsd = encode_chunk(
            k_chunk,
            root=root,
            tensor_name="k",
            layer=layer,
            setting=setting,
            state=states["k"],
            device=device,
            batch_size=batch_size,
            guidance=k_guide,
            curvature=k_curv,
        )

        v_guide = v_curv = attn = None
        if setting.method == "attn_output" and scope_enabled(setting, "v"):
            v_guide, v_curv, attn = v_guidance(q_sample, k_chunk.to(device=device), v_output_error)
        v_hat_bhsd = encode_chunk(
            v_chunk,
            root=root,
            tensor_name="v",
            layer=layer,
            setting=setting,
            state=states["v"],
            device=device,
            batch_size=batch_size,
            guidance=v_guide,
            curvature=v_curv,
        )

        if setting.method == "attn_output":
            k_err = (k_chunk.to(device=device).float() - k_hat_bhsd.float())[0]
            logits_err = torch.bmm(q_sample, k_err.transpose(1, 2)) / math.sqrt(k_err.shape[-1])
            k_logit_error = k_logit_error + logits_err.mean(dim=-1)
            if attn is not None:
                v_err = (v_chunk.to(device=device).float() - v_hat_bhsd.float())[0]
                v_output_error = v_output_error + torch.bmm(attn, v_err)

        k_hat_chunks.append(bhsd_to_thnd(k_hat_bhsd.cpu(), end - start))
        v_hat_chunks.append(bhsd_to_thnd(v_hat_bhsd.cpu(), end - start))

    return torch.cat(k_hat_chunks, dim=0), torch.cat(v_hat_chunks, dim=0)


def topk_overlap(logits: torch.Tensor, logits_hat: torch.Tensor, topk: int) -> float:
    k = min(int(topk), logits.shape[-1])
    ref = torch.topk(logits, k=k, dim=-1).indices
    pred = torch.topk(logits_hat, k=k, dim=-1).indices
    return float((ref[:, :, None] == pred[:, None, :]).any(dim=-1).float().mean().item())


def temporal_stats_by_head(x: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    err = (x.float() - x_hat.float()).reshape(-1, CHUNK_FRAMES, x.shape[1], x.shape[2], x.shape[3])
    err = err.permute(0, 2, 1, 3, 4).reshape(err.shape[0], x.shape[1], -1, x.shape[-1])
    if err.shape[0] < 2:
        inner = torch.zeros(x.shape[1])
    else:
        inner = (err[1:] * err[:-1]).sum(dim=-1).mean(dim=(0, 2)).cpu()
    cumulative = torch.cumsum(err, dim=0)
    cum_norm = torch.sqrt((cumulative * cumulative).mean(dim=(0, 2, 3))).cpu()
    return inner, cum_norm


@torch.no_grad()
def metric_rows(
    payload: dict[str, Any],
    k_hat: torch.Tensor,
    v_hat: torch.Tensor,
    *,
    setting: Setting,
    split: str,
    prompt_id: int,
    layer: int,
    device: torch.device,
    query_sample: int,
    key_sample: int,
    topk: int,
    seed: int,
) -> list[dict[str, Any]]:
    q = payload["Q"].float()
    k = payload["K"].float()
    v = payload["V"].float()
    k_err = k - k_hat.float()
    v_err = v - v_hat.float()
    k_mse = (k_err * k_err).mean(dim=(0, 2, 3))
    v_mse = (v_err * v_err).mean(dim=(0, 2, 3))
    k_cos = F.cosine_similarity(k, k_hat.float(), dim=-1).mean(dim=(0, 2))
    v_cos = F.cosine_similarity(v, v_hat.float(), dim=-1).mean(dim=(0, 2))
    k_inner, k_cum = temporal_stats_by_head(k, k_hat)
    v_inner, v_cum = temporal_stats_by_head(v, v_hat)
    n_tokens = k.shape[0] * k.shape[2]

    rows = []
    for head in range(k.shape[1]):
        row_seed = seed + prompt_id * 1000000 + layer * 10000 + head * 100
        q_idx = sample_indices(n_tokens, query_sample, row_seed)
        kv_idx = sample_indices(n_tokens, key_sample, row_seed + 17)
        q_full = q[:, head].reshape(-1, q.shape[-1]).to(device)
        k_full = k[:, head].reshape(-1, k.shape[-1]).to(device)
        kh_full = k_hat[:, head].reshape(-1, k.shape[-1]).to(device)
        v_full = v[:, head].reshape(-1, v.shape[-1]).to(device)
        vh_full = v_hat[:, head].reshape(-1, v.shape[-1]).to(device)
        q_s = q_full.index_select(0, q_idx.to(device))
        k_s = k_full.index_select(0, kv_idx.to(device))
        kh_s = kh_full.index_select(0, kv_idx.to(device))
        v_s = v_full.index_select(0, kv_idx.to(device))
        vh_s = vh_full.index_select(0, kv_idx.to(device))
        logits = q_s @ k_s.T / math.sqrt(k_s.shape[-1])
        logits_hat = q_s @ kh_s.T / math.sqrt(k_s.shape[-1])
        log_p = F.log_softmax(logits.float(), dim=-1)
        log_q = F.log_softmax(logits_hat.float(), dim=-1)
        p = log_p.exp()
        p_hat = log_q.exp()
        av = p @ v_s.float()
        qkv = p_hat @ vh_s.float()
        rows.append(
            {
                "setting": setting.name,
                "method": setting.method,
                "scope": setting.scope,
                "lambda": setting.lam,
                "beta": setting.beta,
                "split": split,
                "prompt_id": prompt_id,
                "layer": layer,
                "head": head,
                "k_reconstruction_mse": float(k_mse[head].item()),
                "k_reconstruction_cosine": float(k_cos[head].item()),
                "v_reconstruction_mse": float(v_mse[head].item()),
                "v_reconstruction_cosine": float(v_cos[head].item()),
                "qk_logits_mse": float(((logits - logits_hat) ** 2).mean().item()),
                "attention_kl": float((p * (log_p - log_q)).sum(dim=-1).mean().item()),
                "topk_overlap": topk_overlap(logits, logits_hat, topk),
                "qkv_output_mse": float(((av - qkv) ** 2).mean().item()),
                "qkv_output_cosine": float(F.cosine_similarity(av, qkv, dim=-1).mean().item()),
                "k_temporal_error_inner": float(k_inner[head].item()),
                "v_temporal_error_inner": float(v_inner[head].item()),
                "k_cumulative_error_norm": float(k_cum[head].item()),
                "v_cumulative_error_norm": float(v_cum[head].item()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["setting"]), []).append(row)
    metric_names = [
        "k_reconstruction_mse",
        "k_reconstruction_cosine",
        "v_reconstruction_mse",
        "v_reconstruction_cosine",
        "qk_logits_mse",
        "attention_kl",
        "topk_overlap",
        "qkv_output_mse",
        "qkv_output_cosine",
        "k_temporal_error_inner",
        "v_temporal_error_inner",
        "k_cumulative_error_norm",
        "v_cumulative_error_norm",
    ]
    out = []
    for name, group in sorted(groups.items()):
        first = group[0]
        item = {
            "setting": name,
            "method": first["method"],
            "scope": first["scope"],
            "lambda": first["lambda"],
            "beta": first["beta"],
            "rows": len(group),
        }
        for metric in metric_names:
            vals = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = sum(vals) / len(vals)
        out.append(item)
    return out


def select_video_settings(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((row for row in summary if row["setting"] == "baseline"), None)
    if baseline is None:
        return []
    selected = []
    for method in ("inner", "input_comp", "attn_output"):
        candidates = [row for row in summary if row["method"] == method]
        if not candidates:
            continue
        candidates.sort(key=lambda row: float(row["qkv_output_mse_mean"]))
        winner = dict(candidates[0])
        qkv_improvement = (
            float(baseline["qkv_output_mse_mean"]) - float(winner["qkv_output_mse_mean"])
        ) / max(float(baseline["qkv_output_mse_mean"]), 1e-12)
        k_ok = float(winner["k_reconstruction_mse_mean"]) <= 1.05 * float(baseline["k_reconstruction_mse_mean"])
        v_ok = float(winner["v_reconstruction_mse_mean"]) <= 1.05 * float(baseline["v_reconstruction_mse_mean"])
        winner["qkv_improvement"] = qkv_improvement
        winner["passes_gate"] = qkv_improvement >= 0.02 and k_ok and v_ok
        selected.append(winner)
    return selected


def video_command(
    root: Path,
    python_exe: Path,
    row: dict[str, Any],
    num_prompts: int = 3,
    query_sample: int = 96,
) -> str:
    args = [
        str(python_exe),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(root / "prompts_test.txt"),
        "--prompt-start",
        "0",
        "--num-prompts",
        str(num_prompts),
        "--output-dir",
        str(root / "videos_error_feedback" / clean_name(str(row["setting"]))),
        "--seed",
        "42",
        "--setting",
        "kvrvq-i4",
        "--num_frames",
        "173",
        "--num_chunk",
        "11",
        "--num_inference_steps",
        "50",
        "--image_path",
        "none",
        "--pose",
        "w-96",
        "--memory_frames",
        "80",
        "--temporal_context_size",
        "76",
        "--pred_latent_size",
        "4",
        "--offload_text_encoder",
        "--k_rvq_codebook_dir",
        str(root / "sweep_k_l8c64_codebooks"),
        "--v_rvq_codebook_dir",
        str(root / "sweep_v_l8c64_codebooks"),
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
    ]
    if str(row.get("method", "none")) != "none":
        args.extend(
            [
                "--rvq_feedback_method",
                str(row["method"]),
                "--rvq_feedback_tensors",
                str(row["scope"]),
                "--rvq_feedback_lambda",
                str(row["lambda"]),
                "--rvq_feedback_beta",
                str(row["beta"]),
                "--rvq_feedback_query_sample",
                str(query_sample),
            ]
        )
    return " ".join(args)


def write_report(root: Path, summary: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    report = root / "reports" / "error_feedback_l8c64_int4.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# L8C64 INT4 Error-Feedback RVQ",
        "",
        "Offline replay uses fixed L8C64 K/V codebooks and changes only encoder assignment.",
        "",
        "## Top Settings",
        "",
        "| setting | qkv mse | attn KL | K mse | V mse | gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    top = sorted(summary, key=lambda row: float(row["qkv_output_mse_mean"]))[:12]
    selected_names = {row["setting"]: row for row in selected}
    for row in top:
        gate = ""
        if row["setting"] in selected_names:
            gate = "pass" if selected_names[row["setting"]].get("passes_gate") else "best-no-pass"
        lines.append(
            f"| {row['setting']} | {float(row['qkv_output_mse_mean']):.6g} | "
            f"{float(row['attention_kl_mean']):.6g} | "
            f"{float(row['k_reconstruction_mse_mean']):.6g} | "
            f"{float(row['v_reconstruction_mse_mean']):.6g} | {gate} |"
        )
    lines.extend(["", "## Video Candidates", ""])
    if selected:
        for row in selected:
            lines.append(
                f"- {row['setting']}: passes_gate={row.get('passes_gate')} "
                f"qkv_improvement={float(row.get('qkv_improvement', 0.0)):.2%}"
            )
    else:
        lines.append("- No candidates selected.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.output_dir
    prompt_ids = int_list(args.prompt_ids)
    layers = int_list(args.layers)
    settings = make_settings(args)
    device = torch.device(args.device)
    out_dir = root / "metrics" / "error_feedback_l8c64_int4"
    rows: list[dict[str, Any]] = []

    for setting in settings:
        print(f"[setting] {setting}", flush=True)
        for prompt_id in prompt_ids:
            for layer in layers:
                path = layer_path(root, args.split, prompt_id, layer)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                k_hat, v_hat = reconstruct_prompt_layer(
                    payload,
                    root=root,
                    layer=layer,
                    setting=setting,
                    device=device,
                    batch_size=args.batch_size,
                    query_sample=args.query_sample,
                    seed=args.seed + prompt_id * 100000 + layer * 1000,
                )
                rows.extend(
                    metric_rows(
                        payload,
                        k_hat,
                        v_hat,
                        setting=setting,
                        split=args.split,
                        prompt_id=prompt_id,
                        layer=layer,
                        device=device,
                        query_sample=args.query_sample,
                        key_sample=args.key_sample,
                        topk=args.topk,
                        seed=args.seed,
                    )
                )

    write_csv(out_dir / "rows.csv", rows)
    summary = aggregate(rows)
    write_csv(out_dir / "summary.csv", summary)
    selected = select_video_settings(summary)
    write_csv(out_dir / "video_candidates.csv", selected)
    if args.write_video_commands:
        baseline = next((row for row in summary if row["setting"] == "baseline"), None)
        command_rows = ([baseline] if baseline is not None else []) + selected
        commands = [
            video_command(root, args.python, row, num_prompts=3, query_sample=args.query_sample)
            for row in command_rows
        ]
        cmd_path = root / "reports" / "error_feedback_l8c64_int4_video_commands.sh"
        cmd_path.parent.mkdir(parents=True, exist_ok=True)
        cmd_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(commands) + "\n", encoding="utf-8")
    write_report(root, summary, selected)
    print(f"[done] rows={out_dir / 'rows.csv'} summary={out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
