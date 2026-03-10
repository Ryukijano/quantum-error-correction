from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pandas as pd
import pytest

from benchmarks.config import load_spec
from benchmarks.models import suppression_factors
from benchmarks.runner import run_spec


def test_load_spec_yaml_and_json():
    threshold = load_spec("benchmarks/specs/threshold_sweep.yaml")
    repetition = load_spec("benchmarks/specs/repetition_suppression.json")

    assert threshold.benchmark_type == "threshold_sweep"
    assert repetition.benchmark_type == "repetition_suppression"
    assert "distances" in threshold.parameters
    assert "physical_error_rates" in repetition.parameters


def test_suppression_factor_aggregation():
    factors = suppression_factors([1e-1, 1e-2, 2e-3])
    assert factors[0] is None
    assert math.isclose(factors[1], 10.0)
    assert math.isclose(factors[2], 5.0)


# Expected metric columns per benchmark type.  Each set is a subset that must
# be present in the returned DataFrame so that downstream analysis always finds
# the columns it needs.
_SPEC_EXPECTED_COLUMNS = [
    pytest.param(
        "benchmarks/specs/threshold_sweep.yaml",
        {"logical_error_rate", "physical_error_rate", "distance",
         "qubit_overhead", "cycle_count_estimate", "decode_latency_us_estimate"},
        id="threshold_sweep",
    ),
    pytest.param(
        "benchmarks/specs/distance_scaling.yaml",
        {"logical_error_rate", "suppression_factor",
         "qubit_overhead", "cycle_count_estimate", "decode_latency_us_estimate"},
        id="distance_scaling",
    ),
    pytest.param(
        "benchmarks/specs/repetition_suppression.json",
        {"logical_error_rate", "suppression_factor",
         "qubit_overhead", "cycle_count_estimate", "decode_latency_us_estimate"},
        id="repetition_suppression",
    ),
    pytest.param(
        "benchmarks/specs/overhead_comparison.yaml",
        {"assumption", "logical_error_rate",
         "qubit_overhead", "cycle_count_estimate", "decode_latency_us_estimate"},
        id="overhead_comparison",
    ),
]


@pytest.mark.parametrize("spec_path,required_cols", _SPEC_EXPECTED_COLUMNS)
def test_run_spec_outputs_expected_metrics(spec_path, required_cols, tmp_path):
    spec = load_spec(spec_path)
    df = run_spec(spec, output_dir=tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert required_cols.issubset(df.columns)
    assert (tmp_path / f"{spec.output_prefix}.csv").exists()
    assert (tmp_path / f"{spec.output_prefix}.png").exists()


def test_overhead_comparison_has_both_assumptions(tmp_path):
    """overhead_comparison must contain both 'surface' and 'qldpc' assumption rows
    and must produce a bar-plot (the else branch in _save_plot)."""
    spec = load_spec("benchmarks/specs/overhead_comparison.yaml")
    df = run_spec(spec, output_dir=tmp_path)

    assumptions = set(df["assumption"].unique())
    assert "surface" in assumptions, "Missing 'surface' rows in overhead_comparison output"
    assert "qldpc" in assumptions, "Missing 'qldpc' rows in overhead_comparison output"
    # Verify the pivot used by the bar-plot branch produces one column per assumption.
    pivot = df.pivot(index="distance", columns="assumption", values="qubit_overhead")
    assert {"surface", "qldpc"}.issubset(pivot.columns)
    assert (tmp_path / f"{spec.output_prefix}.png").exists()
