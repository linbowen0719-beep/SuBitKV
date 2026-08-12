from pathlib import Path

from scripts.build_rvq_preconditioned_adaptive_report import (
    THEORY_STAGE_FACTOR,
    parse_runtime_logs,
    theory_normalized_mse,
)


def test_theory_rate_distortion_reference() -> None:
    assert abs(theory_normalized_mse(1) - 2 ** (-0.09375)) < 1e-12
    assert abs(theory_normalized_mse(8) - 2 ** (-0.75)) < 1e-12
    assert abs(theory_normalized_mse(16) - 2 ** (-1.5)) < 1e-12
    assert abs(theory_normalized_mse(28) - 2 ** (-2.625)) < 1e-12
    assert abs(theory_normalized_mse(32) - 0.125) < 1e-12
    assert abs(THEORY_STAGE_FACTOR - theory_normalized_mse(1)) < 1e-12


def test_runtime_log_parser_keeps_encode_decode_separate(tmp_path: Path) -> None:
    log = tmp_path / "sample.log"
    log.write_text(
        "\n".join(
            [
                "KV Cache Quantization Time: 0.50 s",
                "KV Cache Dequantization Time for chunk 2: 1.2500 s over 240 calls",
                "KV Cache Quantization Time: 0.75 s",
                "KV Cache Dequantization Time for chunk 3: 2.5000 s over 300 calls",
            ]
        ),
        encoding="utf-8",
    )
    parsed = parse_runtime_logs([log])
    assert parsed["encode_total_s"] == 1.25
    assert parsed["encode_events"] == 2
    assert parsed["decode_total_s"] == 3.75
    assert parsed["decode_calls"] == 540
