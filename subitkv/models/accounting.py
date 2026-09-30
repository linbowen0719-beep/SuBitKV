"""Physical payload accounting from model runtime events."""

from __future__ import annotations

from typing import Any


def summarize_avg2_accounting(
    events: list[dict],
    codebook_storage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Collapse repeated denoise encodes into one 54-layer cache snapshot."""

    supported = {
        "hyworldplay_rvq_ours_avg2_accounting.v2",
        "hyworldplay_rvq_ours_avg2_accounting.v3",
        "hyworldplay_rvq_temporal_accounting.v4",
    }
    snapshots: dict[int, dict[int, dict[str, Any]]] = {}
    event_rows: dict[int, dict[int, dict[str, Any]]] = {}
    for event in events:
        accounting = event.get("post_truncation_accounting")
        if (
            not isinstance(accounting, dict)
            or accounting.get("schema_version") not in supported
        ):
            continue
        spans = int(event.get("history_spans", 0))
        layer = int(event["layer"])
        # Assignment intentionally keeps the last copy: official four-step
        # denoising rebuilds an equivalent history several times.
        snapshots.setdefault(spans, {})[layer] = accounting
        event_rows.setdefault(spans, {})[layer] = event
    if not snapshots:
        return None
    byte_fields = (
        "physical_local_k_bytes",
        "physical_local_v_bytes",
        "physical_residual_k_bytes",
        "physical_residual_v_bytes",
        "physical_rvq_base_k_bytes",
        "physical_rvq_base_v_bytes",
        "physical_rvq_indices_k_bytes",
        "physical_rvq_indices_v_bytes",
        "physical_rvq_scales_k_bytes",
        "physical_rvq_scales_v_bytes",
        "physical_rvq_early_stop_depth_k_bytes",
        "physical_rvq_early_stop_depth_v_bytes",
        "physical_rvq_other_tensor_k_bytes",
        "physical_rvq_other_tensor_v_bytes",
        "physical_shared_k_bytes",
        "physical_shared_v_bytes",
        "physical_payload_bytes",
    )
    codebook_bytes = int((codebook_storage or {}).get("combined_storage_bytes", 0))
    rows = []
    for spans in sorted(snapshots):
        by_layer = snapshots[spans]
        totals = {
            field: sum(int(row.get(field, 0)) for row in by_layer.values())
            for field in byte_fields
        }
        bf16_k_bytes = sum(
            int(
                accounting.get(
                    "bf16_k_bytes", event_rows[spans][layer].get("bf16_bytes", 0) // 2
                )
            )
            for layer, accounting in by_layer.items()
        )
        bf16_v_bytes = sum(
            int(
                accounting.get(
                    "bf16_v_bytes", event_rows[spans][layer].get("bf16_bytes", 0) // 2
                )
            )
            for layer, accounting in by_layer.items()
        )
        bf16_bytes = int(bf16_k_bytes + bf16_v_bytes)
        payload = int(totals["physical_payload_bytes"])
        residual_payload = int(
            totals["physical_residual_k_bytes"]
            + totals["physical_residual_v_bytes"]
            + totals["physical_shared_k_bytes"]
            + totals["physical_shared_v_bytes"]
        )
        scalar_values = int(bf16_bytes // 2)
        nominal_sum = sum(
            int(accounting.get("selected_k_bit_sum", 0))
            + int(accounting.get("selected_v_bit_sum", 0))
            for accounting in by_layer.values()
        )
        k_payload = int(
            totals["physical_local_k_bytes"] + totals["physical_shared_k_bytes"]
        )
        v_payload = int(
            totals["physical_local_v_bytes"] + totals["physical_shared_v_bytes"]
        )
        k_codebook_bytes = int(
            (codebook_storage or {}).get("K", {}).get("codebook_storage_bytes", 0)
        )
        v_codebook_bytes = int(
            (codebook_storage or {}).get("V", {}).get("codebook_storage_bytes", 0)
        )
        rows.append(
            {
                "history_spans": spans,
                "layers": len(by_layer),
                "bf16_k_bytes": bf16_k_bytes,
                "bf16_v_bytes": bf16_v_bytes,
                "bf16_kv_bytes": bf16_bytes,
                **totals,
                "shared_codebook_bytes": codebook_bytes,
                "nominal_residual_avg_bits": (
                    nominal_sum / (2.0 * len(by_layer) * spans)
                    if by_layer and spans
                    else None
                ),
                "physical_residual_bpp": (
                    residual_payload * 8.0 / scalar_values if scalar_values else None
                ),
                "dynamic_kv_bpp": (
                    payload * 8.0 / scalar_values if scalar_values else None
                ),
                "one_video_total_bpp": (
                    (payload + codebook_bytes) * 8.0 / scalar_values
                    if scalar_values
                    else None
                ),
                "two_video_amortized_bpp": (
                    (payload + codebook_bytes / 2.0) * 8.0 / scalar_values
                    if scalar_values
                    else None
                ),
                "compression_ratio_K_excluding_shared_codebook": (
                    bf16_k_bytes / k_payload if k_payload else None
                ),
                "compression_ratio_V_excluding_shared_codebook": (
                    bf16_v_bytes / v_payload if v_payload else None
                ),
                "compression_ratio_K_including_one_codebook_copy": (
                    bf16_k_bytes / (k_payload + k_codebook_bytes)
                    if k_payload + k_codebook_bytes
                    else None
                ),
                "compression_ratio_V_including_one_codebook_copy": (
                    bf16_v_bytes / (v_payload + v_codebook_bytes)
                    if v_payload + v_codebook_bytes
                    else None
                ),
                "compression_ratio_vs_bf16_excluding_shared_codebooks": (
                    bf16_bytes / payload if payload else None
                ),
                "compression_ratio_vs_bf16_including_one_codebook_copy": (
                    bf16_bytes / (payload + codebook_bytes)
                    if payload + codebook_bytes
                    else None
                ),
                "compression_ratio_vs_bf16_two_video_amortized": (
                    bf16_bytes / (payload + codebook_bytes / 2.0)
                    if payload + codebook_bytes
                    else None
                ),
            }
        )
    return {
        "schema_version": "hyworldplay_rvq_avg2_physical_summary.v1",
        "baseline": "BF16 K/V tensor payload",
        "repeated_denoise_rebuilds_deduplicated": True,
        "snapshots": rows,
        "terminal_snapshot": rows[-1],
        "codebook_storage": codebook_storage,
    }
