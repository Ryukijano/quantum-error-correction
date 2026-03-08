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


@pytest.mark.parametrize("real_prob", [np.float32(0.01), np.float64(0.2), 0, 1])
def test_compare_nested_policies_accepts_real_probabilities(real_prob):
    comparison = compare_nested_policies(
        distance=3,
        rounds=3,
        p=real_prob,
        shots=4,
        seed=11,
    )

    for metrics in comparison.values():
        assert metrics["p"] == float(real_prob)


@pytest.mark.parametrize("bad_distance", [2, 4, 1.5])
def test_compare_nested_policies_rejects_invalid_distance(bad_distance):
    with pytest.raises(ValueError, match="distance"):
        compare_nested_policies(distance=bad_distance, rounds=3, p=0.001, shots=8)


@pytest.mark.parametrize("bad_rounds", [0, -1, 1.2])
def test_compare_nested_policies_rejects_invalid_rounds(bad_rounds):
    with pytest.raises(ValueError, match="rounds"):
        compare_nested_policies(distance=3, rounds=bad_rounds, p=0.001, shots=8)


@pytest.mark.parametrize("bad_shots", [0, -5, 2.5])
def test_compare_nested_policies_rejects_invalid_shots(bad_shots):
    with pytest.raises(ValueError, match="shots"):
        compare_nested_policies(distance=3, rounds=3, p=0.001, shots=bad_shots)


@pytest.mark.parametrize("bad_p", [-0.1, 1.1, np.nan, complex(0.5), "0.1", np.array([0.1]), True])
def test_compare_nested_policies_rejects_invalid_p(bad_p):
    with pytest.raises(ValueError, match="p"):
        compare_nested_policies(distance=3, rounds=3, p=bad_p, shots=8)


def test_compare_nested_policies_rejects_non_callable_builders():
    with pytest.raises(ValueError, match="static_builder"):
        compare_nested_policies(distance=3, rounds=3, p=0.001, shots=8, static_builder="not callable")

    with pytest.raises(ValueError, match="dynamic_builder"):
        compare_nested_policies(distance=3, rounds=3, p=0.001, shots=8, dynamic_builder=123)
