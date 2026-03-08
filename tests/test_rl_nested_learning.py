import numpy as np
import pytest

stim = pytest.importorskip("stim")

from surface_code_in_stem.dynamic import (
    hexagonal_surface_code,
    iswap_surface_code,
    walking_surface_code,
)
from surface_code_in_stem.rl_nested_learning import compare_nested_policies, tabulate_comparison


@pytest.mark.parametrize(
    "dynamic_builder",
    [hexagonal_surface_code, iswap_surface_code, walking_surface_code],
    ids=["hexagonal", "iswap", "walking"],
)
def test_compare_nested_policies_and_tabulation(dynamic_builder):
    comparison = compare_nested_policies(
        distance=3,
        rounds=1,
        p=0.001,
        shots=4,
        seed=7,
        dynamic_builder=dynamic_builder,
    )

    assert set(comparison.keys()) == {"static", "dynamic"}
    for metrics in comparison.values():
        assert metrics["distance"] == 3
        assert metrics["rounds"] == 1
        assert metrics["shots"] == 4
        assert 0 <= metrics["logical_error_rate"] <= 1
        assert np.isfinite(metrics["logical_error_rate"])

    rows = list(tabulate_comparison(comparison))
    assert {row["policy"] for row in rows} == {"static", "dynamic"}
    for row in rows:
        assert {"policy", "builder", "logical_error_rate"}.issubset(row.keys())
        assert np.isfinite(row["logical_error_rate"])
