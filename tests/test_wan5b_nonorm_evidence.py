from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/run_wan5b_nonorm_evidence.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("wan5b_nonorm_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_aggregator():
    path = REPO_ROOT / "scripts/aggregate_wan5b_attention_report.py"
    spec = importlib.util.spec_from_file_location("wan5b_attention_report_aggregator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_codebooks(root: Path, *, heads: int = 2, dim: int = 4, mode: str = "none") -> None:
    root.mkdir(parents=True)
    generator = torch.Generator().manual_seed(9)
    for head in range(heads):
        torch.save(
            {
                "codebooks": torch.randn(8, 64, dim, generator=generator) * 0.1,
                "input_norm_mode": mode,
                "config": {
                    "layer": 0,
                    "head": head,
                    "levels": 8,
                    "codebook_size": 64,
                    "input_norm_mode": mode,
                },
            },
            root / f"layer_0_head_{head}.pt",
        )


def write_paired_sample(path: Path, *, heads: int = 2, q_tokens: int = 4, k_tokens: int = 8, dim: int = 4) -> None:
    generator = torch.Generator().manual_seed(4)
    query = torch.randn(1, heads, q_tokens, dim, generator=generator)
    key = torch.randn(1, heads, k_tokens, dim, generator=generator)
    value = torch.randn(1, heads, k_tokens, dim, generator=generator)
    q_cos = torch.ones_like(query)
    q_sin = torch.zeros_like(query)
    k_cos = torch.ones_like(key)
    k_sin = torch.zeros_like(key)
    query_viewmats = torch.eye(4).reshape(1, 1, 4, 4).expand(1, q_tokens, -1, -1).clone()
    key_viewmats = torch.eye(4).reshape(1, 1, 4, 4).expand(1, k_tokens, -1, -1).clone()
    query_Ks = torch.eye(3).reshape(1, 1, 3, 3).expand(1, q_tokens, -1, -1).clone()
    key_Ks = torch.eye(3).reshape(1, 1, 3, 3).expand(1, k_tokens, -1, -1).clone()
    torch.save(
        {
            "query_pre": query,
            "key_pre": key,
            "value_pre": value,
            "query_post_rope": query.clone(),
            "key_post_rope": key.clone(),
            "value_post_rope": value.clone(),
            "query_post_prope": query.clone(),
            "key_post_prope": key.clone(),
            "value_post_prope": value.clone(),
            "query_indices": torch.arange(q_tokens),
            "key_indices": torch.arange(k_tokens),
            "query_rope_cos": q_cos,
            "query_rope_sin": q_sin,
            "key_rope_cos": k_cos,
            "key_rope_sin": k_sin,
            "query_viewmats": query_viewmats,
            "query_Ks": query_Ks,
            "key_viewmats": key_viewmats,
            "key_Ks": key_Ks,
            "metadata": {
                "layer": 0,
                "alignment": "exact_same_call_same_indices",
                "tokens_per_frame": 4,
                "prompt_id": "synthetic",
            },
        },
        path,
    )


def test_camera_prope_matrix_helper_matches_upstream_and_head_specific_gather() -> None:
    runner = load_runner()
    generator = torch.Generator().manual_seed(31)
    q = torch.randn(1, 2, 4, 8, generator=generator)
    k = torch.randn(1, 2, 4, 8, generator=generator)
    v = torch.randn(1, 2, 4, 8, generator=generator)
    viewmats = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 4, -1, -1).clone()
    viewmats[0, :, 0, 3] = torch.tensor([0.0, 0.1, 0.2, 0.3])
    Ks = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 4, -1, -1).clone()
    Ks[0, :, 0, 0] = torch.tensor([1.0, 1.1, 1.2, 1.3])
    Ks[0, :, 1, 1] = torch.tensor([0.9, 1.0, 1.1, 1.2])

    q_ref, k_ref, v_ref, output_fn = runner.prope_qkv(
        q, k, v, viewmats=viewmats, Ks=Ks, patches_x=40, patches_y=22
    )
    q_matrix, kv_matrix, output_matrix = runner.camera_prope_matrices(viewmats, Ks)
    torch.testing.assert_close(runner.apply_camera_prope_matrix(q, q_matrix), q_ref)
    torch.testing.assert_close(runner.apply_camera_prope_matrix(k, kv_matrix), k_ref)
    torch.testing.assert_close(runner.apply_camera_prope_matrix(v, kv_matrix), v_ref)
    torch.testing.assert_close(
        runner.apply_camera_prope_matrix(q, output_matrix), output_fn(q)
    )

    ids = torch.tensor([[[0, 3], [2, 1]]])
    selected_k = runner.gather_tokens(k, ids)
    selected_viewmats = runner.gather_camera_metadata(viewmats, ids)
    selected_Ks = runner.gather_camera_metadata(Ks, ids)
    _, selected_matrix, _ = runner.camera_prope_matrices(selected_viewmats, selected_Ks)
    torch.testing.assert_close(
        runner.apply_camera_prope_matrix(selected_k, selected_matrix),
        runner.gather_tokens(k_ref, ids),
    )


def test_codebook_validation_requires_no_norm_l8c64(tmp_path: Path) -> None:
    runner = load_runner()
    valid = tmp_path / "valid"
    invalid = tmp_path / "invalid"
    write_codebooks(valid)
    write_codebooks(invalid, mode="rms")

    metadata = runner.validate_codebook_dir(valid, layer=0, heads=2, expected_dim=4)
    assert metadata["input_norm_mode"] == "none"
    assert metadata["shape"] == [8, 64, 4]
    with pytest.raises(ValueError, match="input_norm_mode=none"):
        runner.validate_codebook_dir(invalid, layer=0, heads=2, expected_dim=4)


def test_no_norm_prefix_and_physical_logical_storage(tmp_path: Path) -> None:
    runner = load_runner()
    codebooks = tmp_path / "codebooks"
    write_codebooks(codebooks)
    tensor = torch.randn(1, 2, 8, 4)
    packed = runner.encode_fixed(tensor, layer=0, directory=codebooks, batch_size=16)

    runner.assert_no_token_scale(packed)
    assert not (runner.SCALE_KEYS & packed.keys())
    l6 = runner.prefix_packed(packed, 6)
    assert l6["indices"].shape == (6, 1, 2, 8)
    assert torch.allclose(
        runner.decode_prefix(packed, 6, output_dtype=torch.float32),
        runner.rvq_dequantize_tensor_cache(l6, torch.float32),
    )
    storage = runner.payload_storage(packed)
    assert storage["physical_index_tensor_bytes"] == 8 * 1 * 2 * 8
    assert storage["logical_index_bytes"] == 8 * 1 * 2 * 8 * 6 // 8
    assert storage["per_token_scale_bytes"] == 0


def test_tier_schedule_matches_average_depth_six() -> None:
    runner = load_runner()
    assert runner.tier_depths(4) == [8, 4, 4, 8]
    depths = runner.tier_depths(11)
    assert depths == [8, 4, 4, 6, 6, 6, 6, 6, 6, 6, 8]
    assert sum(depths) / len(depths) == 6.0


def test_selected_no_norm_decode_matches_dense_token_gather(tmp_path: Path) -> None:
    runner = load_runner()
    codebooks = tmp_path / "codebooks"
    write_codebooks(codebooks)
    tensor = torch.randn(1, 2, 8, 4)
    packed = runner.encode_fixed(tensor, layer=0, directory=codebooks, batch_size=16)
    token_ids = torch.tensor([[[0, 3, 7], [1, 2, 6]]])
    selected = runner.decode_selected_tokens(packed, token_ids, output_dtype=torch.float32)
    dense = runner.rvq_dequantize_tensor_cache(packed, torch.float32)
    assert torch.allclose(selected, runner.gather_tokens(dense, token_ids))
    runner.assert_no_token_scale(runner.gather_packed_tokens(packed, token_ids))


def test_h1_blockwise_qk_and_streaming_attention_match_explicit_restore(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    k_codebooks = tmp_path / "k"
    v_codebooks = tmp_path / "v"
    write_codebooks(k_codebooks)
    write_codebooks(v_codebooks)
    generator = torch.Generator().manual_seed(41)
    query = torch.randn(1, 2, 5, 4, generator=generator)
    key_pre = torch.randn(1, 2, 11, 4, generator=generator)
    value_pre = torch.randn(1, 2, 11, 4, generator=generator)
    angle = torch.linspace(-0.7, 0.9, 11).reshape(1, 1, 11, 1)
    cos = torch.cos(angle).expand(1, 2, 11, 4).contiguous()
    sin = torch.sin(angle).expand(1, 2, 11, 4).contiguous()
    packed_k = runner.encode_fixed(
        key_pre, layer=0, directory=k_codebooks, batch_size=32
    )
    packed_v = runner.encode_fixed(
        value_pre, layer=0, directory=v_codebooks, batch_size=32
    )

    decoded_k_pre = runner.rvq_dequantize_tensor_cache(packed_k, torch.float32)
    decoded_v = runner.rvq_dequantize_tensor_cache(packed_v, torch.float32)
    decoded_k_post = runner.apply_saved_rope(decoded_k_pre, cos, sin)
    explicit_logits = torch.matmul(query, decoded_k_post.transpose(-1, -2))
    explicit_output = torch.nn.functional.scaled_dot_product_attention(
        query, decoded_k_post, decoded_v
    )

    block_logits = runner.compressed_qk_blockwise_reference(
        query, packed_k, cos, sin, block_tokens=4
    )
    streaming_output = runner.exact_compressed_streaming_attention_reference(
        query,
        packed_k,
        packed_v,
        cos,
        sin,
        block_tokens=4,
        output_dtype=torch.float32,
    )
    assert torch.allclose(block_logits, explicit_logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(streaming_output, explicit_output, atol=2e-6, rtol=2e-6)
    assert (
        runner.relative_l2(streaming_output, explicit_output)
        <= runner.EXACT_STREAMING_RELATIVE_L2_TOLERANCE
    )
    manifest = json.loads(
        (
            REPO_ROOT
            / "experiments/long_video/wan5b_nonorm_attention_report_hypotheses.json"
        ).read_text()
    )
    h1 = next(item for item in manifest["hypotheses"] if item["id"] == "H1")
    assert (
        h1["controlled_experiment"]["validity_gate"]["max"]
        == runner.EXACT_STREAMING_RELATIVE_L2_TOLERANCE
    )
    middle = runner.slice_packed_token_block(packed_k, 3, 8)
    runner.assert_no_token_scale(middle)
    assert torch.equal(middle["indices"], packed_k["indices"][..., 3:8])


def test_legacy_post_rope_sample_disables_accuracy(tmp_path: Path) -> None:
    runner = load_runner()
    sample = tmp_path / "legacy.pt"
    tensor = torch.randn(1, 2, 8, 4)
    torch.save(
        {
            "query": tensor[:, :, :4],
            "key": tensor,
            "value": tensor,
            "metadata": {"layer": 0},
        },
        sample,
    )
    track = runner.load_track(sample, torch.device("cpu"), q_limit=4, k_limit=8)
    assert not track.paired
    assert not track.accuracy_valid
    assert "post-RoPE-only" in track.accuracy_reason


def test_synthetic_cli_writes_evidence_and_run_info(tmp_path: Path) -> None:
    k_dir = tmp_path / "k"
    v_dir = tmp_path / "v"
    sample = tmp_path / "paired.pt"
    output = tmp_path / "results"
    write_codebooks(k_dir)
    write_codebooks(v_dir)
    write_paired_sample(sample)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sample",
            str(sample),
            "--k-codebook-dir",
            str(k_dir),
            "--v-codebook-dir",
            str(v_dir),
            "--output-root",
            str(output),
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--query-tokens",
            "4",
            "--max-key-tokens",
            "8",
            "--page-size",
            "4",
            "--chunk-tokens",
            "4",
            "--sections",
            "H1,H3,H4,H5,H6,H7",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(completed.stdout.strip().splitlines()[-1])
    evidence = json.loads((run_dir / "evidence.json").read_text())
    run_info = json.loads((run_dir / "run_info.json").read_text())
    hypotheses = json.loads((run_dir / "hypotheses.json").read_text())
    track = evidence["tracks"][0]
    assert track["standard_rope_accuracy_valid"] is True
    assert track["packed_invariants"]["k_no_scale"] is True
    assert track["H1"]["nonmaterializing_exact_compressed"]["status"] == "measured"
    assert track["H1"]["nonmaterializing_exact_compressed"]["reference_equivalence_pass"] is True
    assert track["H1"]["standalone_compressed_qk"]["status"] == "measured"
    assert "standalone_compressed_qk" in track["H1"]["timing"]
    assert set(track["H6"]["standard_rope"]["fractions"]["0.2"]) == {
        "naive_pre_rope",
        "position_aware_page_center",
        "post_rope_upper_bound",
    }
    assert track["H6"]["camera_prope"]["status"] == "available"
    assert track["H6"]["camera_prope"]["replay"]["exact_replay"] is True
    assert set(track["H6"]["camera_prope"]["fractions"]["0.2"]) == {
        "naive_pre_rope",
        "position_aware_page_center",
        "post_rope_upper_bound",
    }
    assert track["H7"]["depth_schedule"] == [8, 8]
    assert run_info["input_norm_mode"] == "none"
    assert run_info["per_token_scale"] is False
    assert hypotheses["registered_before_measurement"] is True
    canonical = json.loads((run_dir / "canonical/h6.json").read_text())
    assert canonical["schema_version"] == "wan5b_report_evidence.v1"
    assert canonical["hypothesis_id"] == "H6"
    assert canonical["experiment_id"] == "h6_paired_pre_post_rope_routing"
    assert canonical["baseline_id"] == "nonorm_l8c64_int4"
    assert canonical["status"] == "complete"
    assert canonical["run_info_path"].endswith("run_info.json")
    h1_canonical = json.loads((run_dir / "canonical/h1.json").read_text())
    assert h1_canonical["status"] == "complete"
    aggregator = load_aggregator()
    manifest = json.loads(aggregator.DEFAULT_HYPOTHESES.read_text(encoding="utf-8"))
    synthetic_baseline = dict(manifest["baseline"])
    synthetic_baseline.update(
        {
            "k_codebook_path": str(k_dir.resolve()),
            "v_codebook_path": str(v_dir.resolve()),
            "k_codebook_hash": aggregator.stable_codebook_directory_hash(k_dir),
            "v_codebook_hash": aggregator.stable_codebook_directory_hash(v_dir),
        }
    )
    accepted, rejected, warnings = aggregator.load_external_evidence(
        [run_dir / "canonical/h6.json"], synthetic_baseline
    )
    assert rejected == []
    assert warnings == []
    assert len(accepted) == 1
    assert accepted[0]["experiment_id"] == "h6_paired_pre_post_rope_routing"
