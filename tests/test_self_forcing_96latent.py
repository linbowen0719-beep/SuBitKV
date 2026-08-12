from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from quant_videogen.chunk_policy import completed_chunk_compression_range


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = REPO_ROOT / "scripts" / "Self-Forcing" / "run_96latent_experiment.py"
    spec = importlib.util.spec_from_file_location("self_forcing_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_policy_compresses_each_completed_chunk_once():
    chunks = [3] * 32
    ranges = [completed_chunk_compression_range(index, chunks, "all") for index in range(32)]
    assert ranges == [(index * 3, (index + 1) * 3) for index in range(32)]


def test_first_recent_policy_keeps_first_and_latest_chunks_bf16():
    chunks = [3] * 32
    ranges = [completed_chunk_compression_range(index, chunks, "first_recent") for index in range(32)]
    assert ranges[:2] == [None, None]
    assert ranges[2] == (3, 6)
    assert ranges[-1] == (90, 93)
    compressed = {frame for value in ranges if value for frame in range(*value)}
    assert compressed == set(range(3, 93))


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        completed_chunk_compression_range(0, [3], "last_only")


def test_method_cli_maps_qvg_and_rvq_without_reimplementing_codecs():
    runner = load_runner()
    qvg = type("Args", (), {"method": "qvg_int2", "k_codebooks": None, "v_codebooks": None})()
    assert "triton-nstages-kmeans-int2" in runner.method_args(qvg)

    rvq = type(
        "Args",
        (),
        {"method": "first_last_rvq_l8c64_int4", "k_codebooks": Path("k"), "v_codebooks": Path("v")},
    )()
    values = runner.method_args(rvq)
    assert "--k_rvq" in values and "--v_rvq" in values
    assert values[values.index("--compression_policy") + 1] == "first_recent"

    token = type(
        "Args",
        (),
        {"method": "v_token_l32c64", "k_codebooks": Path("k"), "v_codebooks": Path("v_token")},
    )()
    assert "--compression_policy" not in runner.method_args(token)

    channel = type(
        "Args",
        (),
        {
            "method": "first_recent_v_channel_4x_l8c64",
            "k_codebooks": Path("k"),
            "v_codebooks": Path("v_channel"),
        },
    )()
    channel_values = runner.method_args(channel)
    assert channel_values[channel_values.index("--compression_policy") + 1] == "first_recent"
