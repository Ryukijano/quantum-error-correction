import numpy as np
import pytest

stim = pytest.importorskip("stim")

from surface_code_in_stem.rl_nested_learning import compare_nested_policies, tabulate_comparison


def test_compare_nested_policies_and_tabulation():
    expected_keys = {
        "builder",
        "distance",
        "rounds",
        "p",
        "shots",
        "seed",
        "logical_error_rate",
    }

    comparison = compare_nested_policies(
        distance=3,
        rounds=3,
        p=0.001,
        shots=8,
        seed=7,
    )

    assert set(comparison.keys()) == {"static", "dynamic"}
    for policy in ("static", "dynamic"):
        metrics = comparison[policy]
        assert set(metrics.keys()) == expected_keys
        assert metrics["distance"] == 3
        assert metrics["rounds"] == 3
        assert metrics["shots"] == 8
        # Logical error rate is the fraction of shots where logical observable 0 flipped.
        assert 0 <= metrics["logical_error_rate"] <= 1
        assert np.isfinite(metrics["logical_error_rate"])

    rows = list(tabulate_comparison(comparison))
    assert {row["policy"] for row in rows} == {"static", "dynamic"}
    for row in rows:
        assert set(row.keys()) == {"policy", *expected_keys}
        assert np.isfinite(row["logical_error_rate"])
