from scripts.run_global_stage_allocation import greedy_allocate


def test_greedy_allocation_obeys_prefix_and_budget():
    curves = {
        "a": [10.0, 5.0, 4.0, 3.5],
        "b": [10.0, 9.0, 3.0, 2.0],
    }
    out = greedy_allocate(curves, total_stages=5, min_depth=1, max_depth=4)
    assert out == {"a": 2, "b": 3}
    assert sum(out.values()) == 5


def test_greedy_allocation_validates_budget():
    try:
        greedy_allocate({"a": [2.0, 1.0]}, 3, 1, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid budget was accepted")
