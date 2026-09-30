"""Exercise Wan compatibility fixes without loading its model dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_sdpa_compatibility_path_matches_torch():
    torch = pytest.importorskip("torch")
    source = (
        REPO
        / "third_party/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "sageattn"
    )
    namespace = {"F": torch.nn.functional}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    q, k, v = [torch.randn(1, 2, 4, 8) for _ in range(3)]
    actual = namespace["sageattn"](q, k, v, tensor_layout="HND", is_causal=False)
    torch.testing.assert_close(
        actual, torch.nn.functional.scaled_dot_product_attention(q, k, v)
    )
    with pytest.raises(ValueError, match="HND"):
        namespace["sageattn"](q, k, v, tensor_layout="NHD")


def test_final_budget_check_uses_terminal_instead_of_peak():
    source = (
        REPO
        / "third_party/HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_apply_temporal_adaptive_residual_policy"
    )
    assignment = next(
        n
        for n in function.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "terminal_capacity" for t in n.targets
        )
    )
    check = function.body[function.body.index(assignment) + 1]
    code = compile(
        ast.Module(body=[assignment, check], type_ignores=[]), str(source), "exec"
    )
    namespace = dict(
        profile=SimpleNamespace(
            joint_schedule_terminal_bit_chunks=5, residual_capacity_by_step=None
        ),
        self=SimpleNamespace(
            _quant_config=SimpleNamespace(
                temporal_adaptive_allocation_mode="frozen_joint_schedule"
            )
        ),
        capacity=14,
        step=10,
        target=[1] * 5 + [0] * 5,
    )
    exec(code, namespace)
    namespace["target"] = [1] * 4 + [0] * 6
    with pytest.raises(RuntimeError, match="4/5"):
        exec(code, namespace)
    # A legacy online profile may omit capacity, causing its loaded default
    # to be 40. Its terminal check must still use the actual runtime budget.
    namespace["profile"].joint_schedule_terminal_bit_chunks = 40
    namespace["self"]._quant_config.temporal_adaptive_allocation_mode = "optimal"
    namespace["target"] = [2] * 4 + [1] * 6
    exec(code, namespace)
