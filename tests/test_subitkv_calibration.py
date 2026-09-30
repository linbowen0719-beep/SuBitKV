"""Numerical tests of public teacher calibration, without model checkpoints."""

import json

import pytest

torch = pytest.importorskip("torch")

from subitkv.pipeline import live_calibration as live  # noqa: E402
from subitkv.pipeline.common import REPO, load_config, write_json  # noqa: E402


def bundle(tmp_path):
    from subitkv.routed_gaussian import build_fixed_gaussian_tree, ROUTING_SCHEMA
    from subitkv.progressive_sign_adaptive_absmax_scalar import (
        build_midpoint_params,
    )

    root = tmp_path / "bundle"
    root.mkdir()
    tree = build_fixed_gaussian_tree()
    torch.save({"K": tree, "V": tree}, root / "tree_params.pt")
    torch.save(build_midpoint_params(method="midpoint_msb"), root / "v_params.pt")
    write_json(
        root / "routing.json",
        dict(
            schema_version=ROUTING_SCHEMA,
            num_layers=1,
            num_heads=2,
            offsets={"K": [[0, 0]], "V": [[0, 0]]},
        ),
    )
    return root


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_real_residual_codecs_all_prefixes(tmp_path, monkeypatch, device):
    cfg = load_config(REPO / "configs/subitkv.yaml", "self_forcing", "smoke")[
        "calibration"
    ]
    root = bundle(tmp_path)
    source = torch.randn(1, 2, 8, 128, device=device).to(torch.bfloat16)
    monkeypatch.setattr(live, "base", lambda source, *args: torch.zeros_like(source))
    for name in ("K", "V"):
        restored = live.candidates(source, 0, name, cfg, root, spatial=2, span_tokens=4)
        assert set(restored) == {8, 4, 3, 2, 1, 0}
        assert torch.equal(restored[0], torch.zeros_like(source))
        assert all(
            x.shape == source.shape and torch.isfinite(x).all()
            for x in restored.values()
        )
        assert (
            source.float() - restored[8].float()
        ).square().mean() < source.float().square().mean()


def test_live_attention_records_exact_k_v_damage_and_preserves_rng(
    tmp_path, monkeypatch
):
    cfg = load_config(REPO / "configs/subitkv.yaml", "self_forcing", "smoke")
    cfg["model_config"].update(
        spatial_tokens=2, latent_chunk_frames=2, calibration_history_steps=2
    )
    settings = dict(
        config=cfg,
        phase="weights",
        prompt_id=0,
        output=str(tmp_path / "events"),
        bundle=str(tmp_path),
    )
    monkeypatch.setenv("SUBITKV_CALIBRATION_SETTINGS", "enabled")
    monkeypatch.setattr(live, "_SETTINGS", settings)
    monkeypatch.setattr(live, "_SEEN", set())

    def exact_and_zero(source, *args):
        # Deliberately draw from global RNG to test isolation of calibration.
        torch.rand(10)
        return {
            8: source,
            4: source,
            3: source,
            2: source,
            1: source,
            0: torch.zeros_like(source),
        }

    monkeypatch.setattr(live, "candidates", exact_and_zero)
    q, k, v = [torch.randn(1, 2, 4, 128) for _ in range(3)]
    originals = [x.clone() for x in (q, k, v)]
    rng = torch.get_rng_state().clone()
    live.observe(q, k, v, layer=0, transform=lambda a, b: (a, b))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip((q, k, v), originals))
    event = json.loads((tmp_path / "events/layer_0_span_1.json").read_text())
    for name in ("K", "V"):
        assert event["distortions"][name]["8"]["loss"] == 0
        assert event["distortions"][name]["0"]["loss"] > 0
    assert event["distortions"]["V"]["0"]["output_nmse"] == pytest.approx(1)
    assert event["age_weights"] == pytest.approx([1])
    live.observe(
        q, k, v, layer=0, transform=lambda *args: pytest.fail("duplicate event")
    )


def test_live_residual_collects_complete_frame_means(tmp_path, monkeypatch):
    cfg = load_config(REPO / "configs/subitkv.yaml", "self_forcing", "smoke")
    cfg["model_config"].update(
        spatial_tokens=2, latent_chunk_frames=2, calibration_history_steps=1
    )
    cfg["calibration"]["residual"]["samples_per_head"] = 4
    monkeypatch.setenv("SUBITKV_CALIBRATION_SETTINGS", "enabled")
    monkeypatch.setattr(
        live,
        "_SETTINGS",
        dict(
            config=cfg,
            phase="residual",
            prompt_id=0,
            output=str(tmp_path),
            bundle=str(tmp_path),
        ),
    )
    monkeypatch.setattr(live, "_SEEN", set())
    monkeypatch.setattr(live, "base", lambda source, *args: torch.zeros_like(source))
    source = torch.ones(1, 2, 4, 128, dtype=torch.bfloat16)
    live.observe(source, source, source, layer=0, transform=lambda a, b: (a, b))
    saved = torch.load(tmp_path / "layer_0.pt", weights_only=False)
    assert torch.count_nonzero(saved["K_centered"]) == 0
    assert torch.all(saved["V_residual"] == 1)


def test_rvq_base_uses_real_no_norm_codebooks(tmp_path):
    from subitkv.cache_budget import rvq_schedule_budget
    from subitkv.rvq import rvq_encode_tensor_cache

    cfg = load_config(REPO / "configs/subitkv.yaml", "self_forcing", "smoke")[
        "calibration"
    ]
    generator = torch.Generator().manual_seed(917)
    for name in ("k", "v"):
        directory = tmp_path / f"nonorm_{name}_l16c64_codebooks"
        directory.mkdir()
        for head in range(2):
            codebooks = torch.randn(16, 64, 128, generator=generator) * 0.05
            torch.save(
                dict(
                    codebooks=codebooks,
                    input_norm_mode="none",
                    config=dict(input_norm_mode="none", levels=16, codebook_size=64),
                ),
                directory / f"layer_0_head_{head}.pt",
            )
    source = torch.randn(1, 2, 8, 128, generator=generator).to(torch.bfloat16)
    before = source.clone()
    for name in ("K", "V"):
        decoded = live.base(source, 0, name, cfg, tmp_path)
        assert decoded.shape == source.shape
        assert torch.isfinite(decoded).all()
        assert decoded.abs().sum() > 0
    # Verify the byte reservation against the actual index/depth tensors at
    # maximum depth. C64 still uses uint8 indices in the deployed codec.
    state = rvq_encode_tensor_cache(
        source,
        layer_idx=0,
        codebook_dir=tmp_path / "nonorm_k_l16c64_codebooks",
        batch_size=cfg["assignment_batch_size"],
        compute_dtype="bf16",
        codebook_storage_dtype="int4",
        scale_storage_dtype="uint4",
        relative_stop=True,
        relative_stop_threshold=cfg["early_stop"]["threshold"],
        relative_stop_min_depth=16,
        relative_stop_max_depth=16,
    )
    config = dict(
        calibration=cfg,
        model_config=dict(
            heads=2, spatial_tokens=8, latent_chunk_frames=1, head_dim=128
        ),
    )
    reservation = rvq_schedule_budget(config, 0)
    actual_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in state.values()
        if torch.is_tensor(tensor)
    )
    assert state["indices"].dtype == torch.uint8
    assert actual_bytes == reservation.base_bytes_by_step[0]
    assert torch.equal(source, before)
