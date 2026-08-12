import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VIDEO = load_script(
    "run_adaptive_rvq_best_residual_video_bench_perchunk_test",
    "scripts/run_adaptive_rvq_best_residual_video_bench.py",
)
EXACT = load_script(
    "evaluate_qv_int4_perchunk_comparators_test",
    "scripts/evaluate_qv_int4_perchunk_comparators.py",
)


def command(method: str) -> list[str]:
    args = SimpleNamespace(output_root=Path("/tmp/qv-perchunk"))
    return VIDEO.generation_command(args, method, 3)


def value_after(command: list[str], flag: str) -> str:
    assert command.count(flag) == 1
    return command[command.index(flag) + 1]


def test_direct_quantvideo_int4_is_one_chunk_per_compression() -> None:
    cmd = command("direct_qv_int4")
    assert value_after(cmd, "--setting") == "int4"
    assert value_after(cmd, "--kv_quant_factor") == "1"
    assert "--rvq_residual_quantize" not in cmd


def test_direct_quantvideo_int2_is_one_chunk_and_decodes_as_int2() -> None:
    cmd = command("direct_qv_int2")
    assert value_after(cmd, "--setting") == "int2"
    assert value_after(cmd, "--kv_quant_factor") == "1"
    assert "--rvq_residual_quantize" not in cmd
    assert EXACT.quantvideo_bits("triton-nstages-kmeans-int2") == 2
    assert EXACT.quantvideo_bits("triton-nstages-kmeans-int4") == 4


def test_rvq_plus_quantvideo_int4_is_one_chunk_per_compression() -> None:
    cmd = command("rvq_l8_qv_int4")
    assert value_after(cmd, "--setting") == "kvrvq-i4"
    assert (
        value_after(cmd, "--quant_type")
        == "triton-nstages-kmeans-int4"
    )
    assert value_after(cmd, "--rvq_residual_codec") == "quantvideo"
    assert value_after(cmd, "--rvq_num_levels") == "8"
    assert value_after(cmd, "--kv_quant_factor") == "1"
    assert "--rvq_residual_quantize" in cmd


def test_exact_protocol_has_ten_independent_physical_chunks() -> None:
    assert EXACT.QUANTIZED_CHUNKS == 10
    assert EXACT.FRAMES_PER_CHUNK == 4
    assert EXACT.TOKENS_PER_CHUNK == 3520
