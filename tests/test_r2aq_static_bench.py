from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_r2aq_static_bench.py"
SPEC = importlib.util.spec_from_file_location("run_r2aq_static_bench", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def _prompt_split(path: Path, count: int = 20) -> None:
    path.write_text(
        json.dumps(
            {
                "train": [
                    {"prompt_id": index, "prompt": f"train prompt {index}"}
                    for index in range(count)
                ],
                "test": [
                    {"prompt_id": index, "prompt": f"test prompt {index}"}
                    for index in range(10)
                ],
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path, **updates: Any) -> SimpleNamespace:
    values = {
        "command": "run",
        "data_root": tmp_path / "data",
        "output_root": tmp_path / "out",
        "split": "train",
        "prompts": "0:1",
        "layers": "0:1",
        "heads": "0:1",
        "units_per_prompt": 1,
        "ablations": ("scale_waterfill",),
        "key_sample": 6,
        "query_sample": 2,
        "key_sampling": "inference_span",
        "device": "cpu",
        "rvq_assignment_mode": "fp32_proxy",
        "seed": 42,
        "codec_seed": 20260724,
        "n_clusters": 256,
        "kmeans_iters": 2,
        "predictor_group_size": 128,
        "suffix_group_size": 64,
        "selection_fraction": 0.434091,
        "coherent_tail_group_fraction": 0.55,
        "age_bins": 5,
        "utility_weights": None,
        "tree_params": None,
        "shard_index": 0,
        "num_shards": 1,
        "allow_partial": True,
        "skip_qv_reference": True,
        "bootstrap_samples": 100,
    }
    values.update(updates)
    return SimpleNamespace(**values)


class _FakeR2AQ:
    def __init__(self) -> None:
        self.encode_calls: list[tuple[str, str]] = []
        self.encode_kwargs: list[dict[str, Any]] = []
        self.decode_dtypes: list[torch.dtype] = []

    def encode_r2aq_residual(
        self,
        tensor: torch.Tensor,
        *,
        tensor_name: str,
        ablation: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.encode_calls.append((tensor_name, ablation))
        self.encode_kwargs.append(kwargs)
        return {
            "source": tensor.clone(),
            "ablation": ablation,
            "bits": 5,
            "selected_fraction": 0.5,
        }

    def truncate_r2aq_state(
        self, state: Mapping[str, Any], *, nominal_bits: int
    ) -> dict[str, Any]:
        result = dict(state)
        result["bits"] = int(nominal_bits)
        return result

    def decode_r2aq_residual(
        self, state: Mapping[str, Any], *, output_dtype: torch.dtype
    ) -> torch.Tensor:
        self.decode_dtypes.append(output_dtype)
        return state["source"].to(output_dtype)

    def payload_nbytes(self, state: Mapping[str, Any]) -> int:
        return 100 + int(state["bits"]) * 20

    def payload_breakdown(self, state: Mapping[str, Any]) -> dict[str, int]:
        bits = int(state["bits"])
        return {
            "centroid_bytes": 40,
            "label_bytes": 20,
            "scale_bytes": 10,
            "plane_bytes": bits * 20,
            "selector_bytes": 0,
            "other_payload_bytes": 30,
        }


def test_static_ablation_routes_only_intended_shared_parameters(
    tmp_path: Path,
) -> None:
    fake = _FakeR2AQ()
    args = _args(
        tmp_path,
        utility_weights="utility.pt",
        tree_params="tree.pt",
    )
    residual = torch.zeros((1, 1, 8, 128), dtype=torch.bfloat16)
    ages = torch.zeros(8, dtype=torch.long)
    for ablation in (
        "int4_predictor_qv",
        "uniform_bonus",
        "scale_waterfill",
        "utility_waterfill",
        "full_tree",
        "coherent_tail",
        "coherent_tail_g128",
    ):
        bench.encode_r2aq(
            fake,
            residual,
            tensor_name="K",
            ablation=ablation,
            layer=0,
            ages=ages,
            args=args,
        )
    assert [
        call["utility_weights_path"] for call in fake.encode_kwargs
    ] == [
        None,
        None,
        None,
        "utility.pt",
        "utility.pt",
        "utility.pt",
        "utility.pt",
    ]
    assert [
        call["tree_params_path"] for call in fake.encode_kwargs
    ] == [
        None,
        None,
        None,
        None,
        "tree.pt",
        "tree.pt",
        "tree.pt",
    ]
    assert fake.encode_kwargs[-1][
        "coherent_tail_group_fraction"
    ] == pytest.approx(0.55)


def test_balanced_unit_schedule_and_age_bins() -> None:
    units = bench.selected_units(
        list(range(20)), list(range(30)), list(range(24)), 2
    )
    assert len(units) == 40
    assert len(set(units)) == 40
    assert units[:4] == [(0, 0, 0), (0, 11, 5), (1, 7, 13), (1, 18, 18)]
    ages = bench.token_age_bins(10, 5, torch.device("cpu"))
    assert ages.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    sampled = bench.sampled_token_age_bins(
        torch.tensor([9, 0, 5]), 10, 5, torch.device("cpu")
    )
    assert sampled.tolist() == [4, 0, 2]


def test_rvq_payload_accounts_uint8_indices_and_packed_codebooks() -> None:
    payload = bench.rvq_payload(7040)
    assert payload["rvq_index_payload_bytes"] == 2 * 7040 * 8
    # Per K/V: 8*64*128 INT4 values plus 8*64 UINT4 scales.
    expected_codebook = 2 * ((8 * 64 * 128) // 2 + (8 * 64) // 2)
    assert payload["rvq_codebook_payload_bytes"] == expected_codebook
    assert payload["rvq_payload_bytes"] == (
        payload["rvq_index_payload_bytes"]
        + payload["rvq_codebook_payload_bytes"]
    )


def test_attention_metrics_are_finite_and_include_tail_diagnostics() -> None:
    generator = torch.Generator().manual_seed(3)
    q = torch.randn(3, bench.DIM, generator=generator)
    k = torch.randn(7, bench.DIM, generator=generator)
    v = torch.randn(7, bench.DIM, generator=generator)
    metrics = bench.offline_attention_metrics(q, k, v, k.clone(), v.clone())
    assert set(bench.METRIC_NAMES) <= set(metrics)
    assert metrics["K_MSE"] == 0.0
    assert metrics["V_MSE"] == 0.0
    assert metrics["QKV_output_MSE"] == 0.0
    assert metrics["attention_top1_agreement"] == 1.0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_residual_distribution_reports_shape_and_tail_statistics() -> None:
    generator = torch.Generator().manual_seed(9)
    residual = torch.randn(1, 1, 512, bench.DIM, generator=generator)
    metrics = bench.residual_distribution_metrics(residual)
    assert set(metrics) == set(bench.RAW_DISTRIBUTION_METRICS)
    assert 100.0 < metrics["channel_effective_rank"] <= bench.DIM
    assert 0.0 < metrics["channel_top16_energy_fraction"] < 1.0
    assert metrics["abs_P999"] > metrics["abs_P99"] > 0.0
    assert abs(metrics["skew"]) < 0.1


def test_static_unit_encodes_max_stream_once_and_decodes_all_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    layer_file = data_root / "kv_dump" / "train" / "0" / "layer_0.pt"
    layer_file.parent.mkdir(parents=True)
    generator = torch.Generator().manual_seed(4)
    torch.save(
        {
            name: torch.randn(
                2, 1, 4, bench.DIM, generator=generator
            ).to(torch.bfloat16)
            for name in ("Q", "K", "V")
        },
        layer_file,
    )
    args = _args(tmp_path)
    fake = _FakeR2AQ()
    monkeypatch.setattr(bench, "codec_module", lambda: fake)
    monkeypatch.setattr(
        bench.strict_rvq,
        "load_int4_base_codebook",
        lambda *_args, **_kwargs: torch.empty(0),
    )
    monkeypatch.setattr(
        bench.strict_rvq,
        "rvq_residual",
        lambda _args, source, _codebook, _device: (
            torch.zeros_like(source, dtype=torch.bfloat16),
            source.to(torch.bfloat16),
        ),
    )
    rows = bench.static_unit(
        args, prompt=0, layer=0, head=0, device=torch.device("cpu")
    )
    assert fake.encode_calls == [
        ("K", "scale_waterfill"),
        ("V", "scale_waterfill"),
    ]
    assert fake.decode_dtypes == [torch.bfloat16] * 8
    assert [int(row["bits"]) for row in rows] == [0, 1, 2, 3, 4]
    assert all(
        row["attention_scope"] == bench.ATTENTION_PROXY_SCOPE
        for row in rows
    )
    assert all(
        row["QKV_output_MSE"] == pytest.approx(0.0)
        for row in rows[1:]
    )
    # K and V states each spend 100 + B*20 bytes.
    assert [row["residual_payload_bytes"] for row in rows[1:]] == [
        240,
        280,
        320,
        360,
    ]
    bench.validate_rows(rows, args, {(0, 0, 0)})


def test_payload_breakdown_rejects_overcount() -> None:
    class Bad:
        @staticmethod
        def payload_nbytes(_state: Mapping[str, Any]) -> int:
            return 10

        @staticmethod
        def payload_breakdown(_state: Mapping[str, Any]) -> dict[str, int]:
            return {"centroid_bytes": 11}

    with pytest.raises(ValueError, match="larger than payload"):
        bench.state_payload(Bad, {})


def test_writers_refuse_overwrite(tmp_path: Path) -> None:
    json_path = tmp_path / "value.json"
    bench.atomic_json(json_path, {"first": 1})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        bench.atomic_json(json_path, {"second": 2})
    csv_path = tmp_path / "value.csv"
    bench.write_csv(csv_path, [{"x": 1}])
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        bench.write_csv(csv_path, [{"x": 2}])


def test_aggregate_uses_equal_weight_prompt_macro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(
        tmp_path,
        command="aggregate",
        prompts="0:2",
        units_per_prompt=1,
    )
    args.data_root.mkdir(parents=True)
    _prompt_split(args.data_root / "prompt_split.json")
    row_dir = args.output_root / "rows"
    row_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for prompt, layer, head in bench.selected_units(
        [0, 1], [0], [0], 1
    ):
        common = {
            "split": "train",
            "prompt_id": prompt,
            "layer": layer,
            "head": head,
            "key_tokens": 1,
            "query_tokens": 1,
            "attention_scope": bench.ATTENTION_PROXY_SCOPE,
            "attention_runtime_faithful": False,
            "source_tensor_protocol": "test",
            "rvq_assignment_mode": "fp32_proxy",
            "reconstruction_dtype": "test",
        }
        rvq = bench.rvq_payload(1)
        for method, ablation, bits, residual_bytes, mse in (
            ("l8c64_int4_base", "rvq_base", 0, 0, 10.0),
            ("r2aq_scale_waterfill_b1", "scale_waterfill", 1, 10, 1.0),
            ("r2aq_scale_waterfill_b2", "scale_waterfill", 2, 20, 2.0),
            ("r2aq_scale_waterfill_b3", "scale_waterfill", 3, 30, 3.0),
            ("r2aq_scale_waterfill_b4", "scale_waterfill", 4, 40, 4.0),
        ):
            metrics = {name: 0.0 for name in bench.METRIC_NAMES}
            metrics["QKV_output_MSE"] = mse + prompt * 2
            rows.append(
                {
                    **common,
                    "method": method,
                    "ablation": ablation,
                    "bits": bits,
                    "selected_fraction": 0.0,
                    **bench._payload_fields(1000, residual_bytes, rvq),
                    **metrics,
                }
            )
    bench.write_csv(row_dir / "shard_000.csv", rows)
    monkeypatch.setattr(bench, "write_run_info", lambda *_args, **_kwargs: None)
    bench.aggregate(args)
    with (
        args.output_root / "aggregate" / "summary.csv"
    ).open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    candidate = next(
        row
        for row in summary
        if row["method"] == "r2aq_scale_waterfill_b1"
    )
    assert float(candidate["QKV_output_MSE"]) == pytest.approx(2.0)
    payload = json.loads(
        (args.output_root / "aggregate" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["attention_scope"] == bench.ATTENTION_PROXY_SCOPE
    assert "not exact online attention replay" in payload["warning"]
