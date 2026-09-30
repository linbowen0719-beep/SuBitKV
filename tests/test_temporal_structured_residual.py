from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import subitkv.structured_truncatable as structured
from subitkv.temporal_structured_residual import (
    FORMAL_PENDING_BF16_BYTES,
    FORMAL_PHYSICAL_D128_SHARED_CODEBOOK_BYTES,
    FORMAL_UNIFORM_COMPRESSED_SPAN_BYTES,
    FORMAL_V_PLANE_ACROSS_30_LAYERS_BYTES,
    STRUCTURED_TEMPORAL_KV_POLICY,
    STRUCTURED_TEMPORAL_KV_POLICY_31114,
    STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE,
    apply_structured_temporal_cache_updates,
    normalize_structured_temporal_policy,
    plan_structured_temporal_kv_updates,
    structured_temporal_depth_schedule,
    structured_temporal_formal_payload_timeline,
    structured_temporal_storage_audit,
)


class _PackedSpanCache:
    def __init__(self) -> None:
        self.quantized_spans: list[dict] = []

    def append(
        self,
        generation_span: int,
        packed: dict,
        *,
        span_frames: int = 1,
    ) -> None:
        self.quantized_spans.append(
            {
                "start_chunk": int(generation_span) * int(span_frames),
                "end_chunk": (int(generation_span) + 1) * int(span_frames),
                "quant_data": packed,
            }
        )


def _gaussian_params() -> dict:
    thresholds = []
    centroids = []
    for depth in range(4):
        nodes = 1 << depth
        thresholds.append(torch.linspace(-1.5, 1.5, nodes))
        centroids.append(torch.linspace(-2.0, 2.0, 2 * nodes))
    values = {"thresholds": thresholds, "centroids": centroids}
    return {
        "schema_version": "truncatable_residual_params.v1",
        "schemes": {"gaussian_tree": {"K": values, "V": values}},
    }


def _install_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_predictor(flattened, *, n_clusters, max_iters):
        batches, tokens, dim = flattened.shape
        ids = (
            torch.arange(tokens, device=flattened.device)
            .remainder(n_clusters)
            .expand(batches, -1)
            .contiguous()
        )
        centers = torch.zeros(
            batches,
            n_clusters,
            dim,
            dtype=flattened.dtype,
            device=flattened.device,
        )
        sizes = torch.zeros(
            batches, n_clusters, dtype=torch.int64, device=flattened.device
        )
        sizes.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int64))
        return ids, centers, sizes, max_iters

    monkeypatch.setattr(
        structured, "_online_vector_predictor", fake_predictor
    )


def _base(tokens: int) -> dict:
    return {
        "quant_method": "rvq",
        "indices": torch.zeros((8, 1, 1, tokens), dtype=torch.uint8),
        "num_levels": 8,
        "input_norm_mode": "none",
        "codebook_storage_dtype": "int4",
        "rvq_format": "standard",
        "progressive": False,
    }


def _packed(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    bits: int,
) -> dict:
    residual = structured.encode_structured_residual(
        tensor,
        bits=bits,
        tensor_name=tensor_name,
        params_path=_gaussian_params(),
        group_size=128,
        n_clusters=16,
        kmeans_max_iters=2,
        hadamard=True,
        scale_refine_iters=1,
    )
    return {
        "quant_method": "rvq_structured_truncatable_residual",
        "base": _base(int(tensor.shape[2])),
        "residual": residual,
    }


@pytest.mark.parametrize(
    ("policy", "terminal_v"),
    (
        (STRUCTURED_TEMPORAL_KV_POLICY, [4, 1, 1, 1, 3]),
        (STRUCTURED_TEMPORAL_KV_POLICY_31114, [3, 1, 1, 1, 4]),
        (
            STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE,
            [3, 1, 1, 1, 4],
        ),
    ),
)
def test_schedule_and_terminal_payload_exactly_match_uniform_b2(
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    terminal_v: list[int],
) -> None:
    _install_predictor(monkeypatch)
    assert normalize_structured_temporal_policy("structured_temporal") == (
        STRUCTURED_TEMPORAL_KV_POLICY
    )
    assert structured_temporal_depth_schedule(5, tensor_name="K") == [
        2,
        2,
        2,
        2,
        2,
    ]
    assert structured_temporal_depth_schedule(
        5, tensor_name="V", policy=policy
    ) == terminal_v

    k_cache = _PackedSpanCache()
    v_cache = _PackedSpanCache()
    generator = torch.Generator().manual_seed(20260723)
    first_v_full_planes: tuple[torch.Tensor, ...] | None = None
    first_v_full_bytes = 0
    observed_v_plane_deltas: list[int] = []

    for generation_span in range(5):
        tensor = torch.randn(
            (1, 1, 16, 128),
            generator=generator,
            dtype=torch.bfloat16,
        )
        k_cache.append(
            generation_span,
            _packed(tensor, tensor_name="K", bits=2),
        )
        v_cache.append(
            generation_span,
            _packed(tensor, tensor_name="V", bits=4),
        )
        if generation_span == 0:
            first_v = v_cache.quantized_spans[0]["quant_data"]["residual"]
            first_v_full_planes = tuple(first_v["suffix"]["planes"])
            first_v_full_bytes = structured.payload_nbytes(first_v)

        all_updates = plan_structured_temporal_kv_updates(
            (("K", k_cache), ("V", v_cache)),
            generation_frames=1,
            num_cache_spans=generation_span + 1,
            policy=policy,
        )
        for updates in all_updates:
            apply_structured_temporal_cache_updates(updates)

        step_audit = structured_temporal_storage_audit(
            (("K", k_cache), ("V", v_cache)), policy=policy
        )
        assert step_audit["k_within_fixed_terminal_suffix_plane_cap"]
        assert step_audit["v_within_fixed_terminal_suffix_plane_cap"]
        observed_v_plane_deltas.append(
            step_audit["v_plane_delta_vs_current_uniform_b2"]
        )

        # Every committed prefix remains directly decodable.
        for span in v_cache.quantized_spans:
            decoded = structured.decode_structured_residual(
                span["quant_data"]["residual"], torch.bfloat16
            )
            assert decoded.shape == tensor.shape

    assert first_v_full_planes is not None
    first_v = v_cache.quantized_spans[0]["quant_data"]["residual"]
    retained_planes = tuple(first_v["suffix"]["planes"])
    assert len(first_v_full_planes) == 4
    assert len(retained_planes) == terminal_v[0]

    # The second V span is physically reduced from four independent planes to
    # one.  Its retained plane is a clone, not a view keeping B4 storage alive.
    second_v = v_cache.quantized_spans[1]["quant_data"]["residual"]
    assert len(second_v["suffix"]["planes"]) == 1
    second_v_original_plane_bytes = int(
        second_v["suffix"]["planes"][0].numel()
        * second_v["suffix"]["planes"][0].element_size()
    )
    assert structured.payload_nbytes(second_v) == (
        first_v_full_bytes - 3 * second_v_original_plane_bytes
    )
    assert second_v["suffix"]["truncated_from_bits"] == 4

    audit = structured_temporal_storage_audit(
        (("K", k_cache), ("V", v_cache)), policy=policy
    )
    assert audit["k_schedule"] == [2, 2, 2, 2, 2]
    assert audit["v_schedule"] == terminal_v
    assert audit["k_plane_units_per_cache"] == 10
    assert audit["v_plane_units_per_cache"] == 10
    assert audit["payload_delta_vs_uniform_kv_b2_bytes"] == 0
    assert audit["terminal_payload_exactly_matches_uniform_kv_b2"] is True
    expected_deltas = (
        [1, 0, -1, -2, 0]
        if policy == STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE
        else [2, 4, 3, 2, 0]
    )
    assert observed_v_plane_deltas == expected_deltas


def test_physical_truncation_clones_retained_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    tensor = torch.randn(
        (1, 1, 16, 128),
        generator=torch.Generator().manual_seed(9),
        dtype=torch.bfloat16,
    )
    cache = _PackedSpanCache()
    for generation_span in range(3):
        cache.append(
            generation_span,
            _packed(tensor, tensor_name="V", bits=4),
        )
    old_planes = tuple(
        cache.quantized_spans[1]["quant_data"]["residual"]["suffix"]["planes"]
    )
    updates = plan_structured_temporal_kv_updates(
        (("V", cache),),
        generation_frames=1,
        num_cache_spans=3,
    )
    apply_structured_temporal_cache_updates(updates[0])
    new_planes = tuple(
        cache.quantized_spans[1]["quant_data"]["residual"]["suffix"]["planes"]
    )
    assert len(old_planes) == 4
    assert len(new_planes) == 1
    assert new_planes[0].data_ptr() != old_planes[0].data_ptr()
    assert torch.equal(new_planes[0], old_planes[0])


@pytest.mark.parametrize(
    "policy",
    (
        STRUCTURED_TEMPORAL_KV_POLICY,
        STRUCTURED_TEMPORAL_KV_POLICY_31114,
        STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE,
    ),
)
def test_formal_axis_uses_five_eight_frame_7040_token_spans(
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    _install_predictor(monkeypatch)
    tensor = torch.zeros((1, 1, 16, 128), dtype=torch.bfloat16)
    k_cache = _PackedSpanCache()
    v_cache = _PackedSpanCache()
    for generation_span in range(5):
        k_cache.append(
            generation_span,
            _packed(tensor, tensor_name="K", bits=2),
            span_frames=8,
        )
        v_cache.append(
            generation_span,
            _packed(tensor, tensor_name="V", bits=4),
            span_frames=8,
        )
        updates = plan_structured_temporal_kv_updates(
            (("K", k_cache), ("V", v_cache)),
            generation_frames=8,
            num_cache_spans=generation_span + 1,
            policy=policy,
        )
        for group in updates:
            apply_structured_temporal_cache_updates(group)
    assert [
        (span["start_chunk"], span["end_chunk"])
        for span in v_cache.quantized_spans
    ] == [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40)]
    audit = structured_temporal_storage_audit(
        (("K", k_cache), ("V", v_cache)), policy=policy
    )
    assert audit["terminal_payload_exactly_matches_uniform_kv_b2"]


def test_illegal_upgrade_preflight_leaves_all_caches_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    tensor = torch.zeros((1, 1, 16, 128), dtype=torch.bfloat16)
    k_cache = _PackedSpanCache()
    v_cache = _PackedSpanCache()
    for generation_span in range(2):
        k_cache.append(
            generation_span,
            _packed(tensor, tensor_name="K", bits=2),
        )
        v_cache.append(
            generation_span,
            _packed(tensor, tensor_name="V", bits=4),
        )
    first_pass = plan_structured_temporal_kv_updates(
        (("K", k_cache), ("V", v_cache)),
        generation_frames=1,
        num_cache_spans=2,
    )
    for updates in first_pass:
        apply_structured_temporal_cache_updates(updates)

    # Corrupt the V anchor to B3.  The two-span target is still B4, so this is
    # a forbidden upgrade.  K appears first in preflight, making this an
    # explicit all-cache atomicity/rollback check.
    v_anchor = v_cache.quantized_spans[0]["quant_data"]
    v_anchor["residual"] = structured.truncate_structured_state(
        v_anchor["residual"], 3
    )
    k_before = tuple(
        k_cache.quantized_spans[0]["quant_data"]["residual"]["suffix"]["planes"]
    )
    k_ptrs_before = tuple(plane.data_ptr() for plane in k_before)
    v_before = structured.payload_nbytes(v_anchor["residual"])
    with pytest.raises(ValueError, match=r"increase B3->B4"):
        plan_structured_temporal_kv_updates(
            (("K", k_cache), ("V", v_cache)),
            generation_frames=1,
            num_cache_spans=2,
        )
    k_after = tuple(
        k_cache.quantized_spans[0]["quant_data"]["residual"]["suffix"]["planes"]
    )
    assert tuple(plane.data_ptr() for plane in k_after) == k_ptrs_before
    assert structured.payload_nbytes(v_anchor["residual"]) == v_before


def test_eval_and_wan_plumbing_defaults_off() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_path = (
        repo_root
        / "experiments"
        / "HY-WorldPlay"
        / "wan"
        / "inference"
        / "pipeline_wan_w_mem_relative_rope.py"
    )
    module = ast.parse(pipeline_path.read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "__call__"
    )
    positional = call.args.args
    defaults = [None] * (len(positional) - len(call.args.defaults)) + list(
        call.args.defaults
    )
    default_map = {
        arg.arg: default for arg, default in zip(positional, defaults)
    }
    assert (
        ast.literal_eval(
            default_map["kv_quant_structured_temporal_policy"]
        )
        == "none"
    )
    generator_source = (
        pipeline_path.parents[1] / "generate.py"
    ).read_text(encoding="utf-8")
    assert '"kv_quant_structured_temporal_policy", "none"' in generator_source
    assert (
        "kv_quant_structured_temporal_policy=(" in generator_source
    )


def test_launcher_freezes_formal_video_axis() -> None:
    from scripts import run_wan5b_structured_temporal_video_smoke as launch
    from subitkv.sim.quant.quantize_config import QuantizeConfig

    args = launch.parse_args([])
    command = launch.generation_command(args, Path("/tmp/unused"))

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert launch.GENERATION_CHUNKS == 11
    assert launch.CACHE_SPANS == 5
    assert launch.LATENT_FRAMES_PER_SPAN == 8
    assert launch.TOKENS_PER_SPAN == 7040
    assert value("--num_chunk") == "11"
    assert value("--num_frames") == "173"
    assert value("--kv_quant_factor") == "2"
    assert value("--kv_reference_frames") == "40"
    assert value("--quant_type") == "triton-nstages-kmeans-int2"
    assert QuantizeConfig().structured_residual_fast_decode is False
    source_names = {
        str(path.relative_to(launch.REPO_ROOT))
        for path in launch.SOURCE_FILES
    }
    assert {
        "subitkv/structured_truncatable.py",
        "subitkv/kernels/triton/structured_truncatable.py",
        "subitkv/truncatable_residual.py",
        "subitkv/rvq.py",
        "subitkv/kv_cache.py",
        "subitkv/uncompress.py",
    }.issubset(source_names)
    eval_source = (
        launch.REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"
    ).read_text(encoding="utf-8")
    assert (
        'quant_config["structured_residual_fast_decode"] = True'
        in eval_source
    )


def test_peak_safe_post_policy_attention_visible_payload_timeline_is_frozen() -> None:
    rows = structured_temporal_formal_payload_timeline(
        policy=STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE
    )
    expected_mib = [
        1237.5,
        583.06640625,
        1820.56640625,
        1011.4453125,
        2248.9453125,
        1439.82421875,
        2677.32421875,
        1868.203125,
        3105.703125,
        2528.61328125,
    ]
    assert [row["candidate_MiB"] for row in rows] == expected_mib
    assert [
        row["v_plane_delta_vs_current_uniform_b2"]
        for row in rows
        if row["chunk_i"] % 2 == 0
    ] == [1, 0, -1, -2, 0]
    candidate_peak = max(row["candidate_MiB"] for row in rows)
    uniform_peak = max(row["uniform_same_chunk_MiB"] for row in rows)
    assert candidate_peak == 3105.703125
    assert uniform_peak == 3260.390625
    assert candidate_peak < uniform_peak
    assert rows[-1]["candidate_MiB"] == rows[-1]["uniform_same_chunk_MiB"]


def test_formal_payload_constants_are_derived_from_frozen_geometry() -> None:
    layers, tensors, heads, dim = 30, 2, 24, 128
    tokens_per_frame, frames_per_span = 880, 8
    tokens = tokens_per_frame * frames_per_span
    levels, codebook_size, predictor_clusters = 8, 64, 256

    pending_bf16 = (
        layers * tensors * heads * (4 * tokens_per_frame) * dim * 2
    )
    rvq_uint8_indices = (
        layers * tensors * levels * heads * tokens
    )
    values = heads * tokens * dim
    structured_per_state_b2 = (
        heads * predictor_clusters * dim * 2
        + heads * tokens
        + (values // 128) * 2
        + 2 * ((values + 7) // 8)
    )
    uniform_span = (
        rvq_uint8_indices
        + layers * tensors * structured_per_state_b2
    )
    one_v_plane = layers * ((values + 7) // 8)
    codebook = (
        ((levels * codebook_size * dim + 1) // 2)
        + levels * codebook_size * 2
    ) * heads * layers * tensors

    assert pending_bf16 == FORMAL_PENDING_BF16_BYTES
    assert uniform_span == FORMAL_UNIFORM_COMPRESSED_SPAN_BYTES
    assert one_v_plane == FORMAL_V_PLANE_ACROSS_30_LAYERS_BYTES
    assert codebook == FORMAL_PHYSICAL_D128_SHARED_CODEBOOK_BYTES
