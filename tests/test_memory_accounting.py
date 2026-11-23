"""Cache footprint includes shared parameters and observed codec workspace."""

import pytest

from subitkv.memory import cache_footprint


def test_complete_memory_components_are_required():
    assert cache_footprint(100, 10, 25) == 135
    assert cache_footprint(100, None, 25) is None
    with pytest.raises(ValueError, match="Negative"):
        cache_footprint(100, 10, -1)


def test_cuda_workspace_captures_temporary_encode_memory_and_preserves_peak():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA measurement")
    from subitkv.memory import reset_workspace_stats, track_workspace, workspace_stats

    reset_workspace_stats()
    source = torch.ones(1024, device="cuda")
    @track_workspace("encode")
    def encode(x):
        scratch = torch.zeros(4 * 1024 * 1024, device="cuda")
        scratch.add_(x[0])
        return x[:1].clone()
    result = encode(source)
    assert result.item() == 1
    stats = workspace_stats()
    assert stats["codec_workspace_mib"] >= 16
    assert stats["codec_workspace_calls"] == 1
    assert stats["codec_process_peak_mib"] >= stats["codec_workspace_mib"]
