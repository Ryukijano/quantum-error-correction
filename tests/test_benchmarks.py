from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pandas as pd

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


def test_run_spec_outputs_expected_metrics(tmp_path):
    spec = load_spec("benchmarks/specs/distance_scaling.yaml")
    df = run_spec(spec, output_dir=tmp_path)

    required_columns = {
        "logical_error_rate",
        "suppression_factor",
        "qubit_overhead",
        "cycle_count_estimate",
        "decode_latency_us_estimate",
    }
    assert required_columns.issubset(df.columns)
    assert isinstance(df, pd.DataFrame)
    assert (tmp_path / f"{spec.output_prefix}.csv").exists()
    assert (tmp_path / f"{spec.output_prefix}.png").exists()
