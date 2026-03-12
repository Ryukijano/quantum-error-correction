import pytest

np = pytest.importorskip("numpy")

stim = pytest.importorskip("stim")

from surface_code_in_stem.dynamic import (
    hexagonal_surface_code,
    iswap_surface_code,
    walking_surface_code,
    xyz2_hexagonal_code,
)
from surface_code_in_stem.rl_nested_learning import compare_nested_policies, tabulate_comparison


@pytest.mark.parametrize(
    "dynamic_builder",
    [hexagonal_surface_code, iswap_surface_code, walking_surface_code, xyz2_hexagonal_code],
)
def test_compare_nested_policies_and_tabulation(dynamic_builder):
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
        rounds=2,
        p=0.001,
        shots=4,
        seed=7,
        dynamic_builder=dynamic_builder,
    )

    assert set(comparison.keys()) == {"static", "dynamic"}
    for policy in ("static", "dynamic"):
        metrics = comparison[policy]
        assert set(metrics.keys()) == expected_keys
        assert metrics["distance"] == 3
        assert metrics["rounds"] == 2
        assert metrics["shots"] == 4
        # Logical error rate is the fraction of shots where logical observable 0 flipped.
        assert 0 <= metrics["logical_error_rate"] <= 1
        assert np.isfinite(metrics["logical_error_rate"])

    rows = list(tabulate_comparison(comparison))
    assert {row["policy"] for row in rows} == {"static", "dynamic"}
    assert {
        row["builder"]
        for row in rows
        if row["policy"] == "dynamic"
    } == {dynamic_builder.__name__}
    for row in rows:
        assert set(row.keys()) == {"policy", *expected_keys}
        assert np.isfinite(row["logical_error_rate"])


def test_compare_nested_policies_is_deterministic_with_fixed_seed():
    first = compare_nested_policies(
        distance=3,
        rounds=3,
        p=0.001,
        shots=16,
        seed=123,
    )
    second = compare_nested_policies(
        distance=3,
        rounds=3,
        p=0.001,
        shots=16,
        seed=123,
    )

    # Compare explicitly deterministic outputs only.
    for policy in ("static", "dynamic"):
        assert first[policy]["logical_error_rate"] == second[policy]["logical_error_rate"]


def test_xyz2_hexagonal_code_builds_detector_circuit():
    circuit = xyz2_hexagonal_code(distance=3, rounds=2, p=0.001)

    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_observables == 1
    assert circuit.num_detectors > 0
    circuit.detector_error_model(decompose_errors=True)
