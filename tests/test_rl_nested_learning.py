import numpy as np
import pytest

stim = pytest.importorskip("stim")

from surface_code_in_stem.rl_nested_learning import compare_nested_policies, tabulate_comparison


def test_compare_nested_policies_and_tabulation():
    comparison = compare_nested_policies(
        distance=3,
        rounds=3,
        p=0.001,
        shots=8,
        seed=7,
    )

    assert set(comparison.keys()) == {"static", "dynamic"}
    for metrics in comparison.values():
        assert metrics["distance"] == 3
        assert metrics["rounds"] == 3
        assert metrics["shots"] == 8
        assert 0 <= metrics["logical_error_rate"] <= 1
        assert np.isfinite(metrics["logical_error_rate"])

    rows = list(tabulate_comparison(comparison))
    assert {row["policy"] for row in rows} == {"static", "dynamic"}
    for row in rows:
        assert {"policy", "builder", "logical_error_rate"}.issubset(row.keys())
        assert np.isfinite(row["logical_error_rate"])

