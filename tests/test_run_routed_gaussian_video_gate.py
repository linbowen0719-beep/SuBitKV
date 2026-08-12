from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import run_routed_gaussian_video_gate as runner


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_video_gate_selects_cluster_free_grid_codec(tmp_path: Path) -> None:
    args = SimpleNamespace(
        codec="uniform_grid_gaussian",
        smoke=True,
        routing=tmp_path / "routing.json",
        tree_params=tmp_path / "tree.pt",
    )
    command = runner.command(args, 4, tmp_path / "out")
    assert _arg(command, "--rvq_residual_codec") == "uniform_grid_gaussian"
    assert _arg(command, "--routed_gaussian_bits") == "4"
    assert _arg(command, "--routed_gaussian_routing_path") == str(args.routing)
    assert _arg(command, "--routed_gaussian_tree_params_path") == str(
        args.tree_params
    )


def test_video_gate_supports_tensor_codec_ablation(tmp_path: Path) -> None:
    args = SimpleNamespace(
        codec="routed_gaussian",
        k_codec="uniform_grid_gaussian",
        v_codec="routed_gaussian",
        smoke=False,
        routing=tmp_path / "routing.json",
        tree_params=tmp_path / "tree.pt",
    )
    command = runner.command(args, 9, tmp_path / "out")
    assert _arg(command, "--rvq_residual_codec") == "routed_gaussian"
    assert _arg(command, "--rvq_k_residual_codec") == "uniform_grid_gaussian"
    assert _arg(command, "--rvq_v_residual_codec") == "routed_gaussian"
