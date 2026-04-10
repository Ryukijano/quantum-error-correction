"""Tests for benchmark decoder contract columns."""

from __future__ import annotations

import csv
import json
import pytest
from pathlib import Path

from scripts.benchmark_decoders import (
    BenchmarkRow,
    _backend_row_metadata,
    _normalise_sampling_backends,
    _write_csv,
)


def test_backend_row_metadata_exposes_chain_and_flags() -> None:
    probe = {"stim": {"enabled": True, "version": "0.0", "details": {}}}
    metadata = _backend_row_metadata("stim", probe, "trace-001")

    assert metadata["backend"] == "stim"
    assert metadata["backend_chain"] == "selected:stim"
    assert metadata["backend_chain_tokens"] == ["requested:stim", "selected:stim"]
    assert metadata["contract_flags"] == "backend_enabled,contract_met"
    assert metadata["profiler_flags"] == "sample_trace_present,trace_chain_recorded"
    assert metadata["sample_trace_id"] == "trace-001"


def test_backend_row_metadata_marks_disabled_backend_contract_state() -> None:
    probe = {"stim": {"enabled": False, "version": "0.0", "details": {}}}
    metadata = _backend_row_metadata("stim", probe, "")

    assert metadata["backend_chain"] == "selected:stim->disabled"
    assert metadata["backend_enabled"] is False
    assert metadata["fallback_reason"] == "backend unavailable"
    assert metadata["sample_trace_id"] is None
    assert metadata["contract_flags"] == "backend_disabled,contract_fallback"


def test_csv_output_includes_backend_chain_metadata_fields(tmp_path: Path) -> None:
    row = BenchmarkRow(
        domain="circuit",
        family="surface",
        decoder="mwpm",
        distance=3,
        physical_error_rate=0.001,
        shots=128,
        metric_name="logical_error_rate",
        metric_value=0.123,
        backend="stim",
        backend_enabled=True,
        backend_version="0.0",
        fallback_reason=None,
        sample_trace_id="trace-csv",
        backend_chain="selected:stim",
        backend_chain_tokens=["requested:stim", "selected:stim"],
        contract_flags="backend_enabled,contract_met",
        profiler_flags="sample_trace_present,trace_chain_recorded",
    )
    output = tmp_path / "rows.csv"
    _write_csv([row], output)

    with output.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        assert "backend_chain_tokens" in reader.fieldnames
        assert "contract_flags" in reader.fieldnames
        assert "profiler_flags" in reader.fieldnames
        written = next(reader)
        assert written["backend_chain"] == "selected:stim"
        assert json.loads(written["backend_chain_tokens"]) == ["requested:stim", "selected:stim"]
        assert written["contract_flags"] == "backend_enabled,contract_met"


def test_normalise_sampling_backends_accepts_valid_tokens() -> None:
    probe = {
        "stim": {"enabled": True, "version": "x", "details": {}},
        "qhybrid": {"enabled": True, "version": "x", "details": {}},
        "cuquantum": {"enabled": False, "version": "x", "details": {}},
        "qujax": {"enabled": False, "version": "x", "details": {}},
        "cudaq": {"enabled": False, "version": "x", "details": {}},
    }
    tokens = _normalise_sampling_backends("qhybrid,stim", probe)
    assert tokens == ["stim", "qhybrid"]


def test_normalise_sampling_backends_rejects_unknown_backend() -> None:
    probe = {
        "stim": {"enabled": True, "version": "x", "details": {}},
        "qhybrid": {"enabled": True, "version": "x", "details": {}},
    }
    with pytest.raises(ValueError):
        _normalise_sampling_backends("stim,unknown_backend", probe)


def test_normalise_sampling_backends_aliases_produce_stable_sweep_order() -> None:
    probe = {
        "qhybrid": {"enabled": True, "version": "x", "details": {}},
        "cuquantum": {"enabled": False, "version": "x", "details": {}},
        "stim": {"enabled": True, "version": "x", "details": {}},
        "cudaq": {"enabled": False, "version": "x", "details": {}},
        "qujax": {"enabled": True, "version": "x", "details": {}},
    }
    expected = ["stim", "qhybrid", "cuquantum", "qujax", "cudaq"]

    assert _normalise_sampling_backends("sweep", probe) == expected
    assert _normalise_sampling_backends("all", probe) == expected
    assert _normalise_sampling_backends("*", probe) == expected


def test_sweep_mode_outputs_preserve_chain_token_shape_for_csv_and_json(tmp_path: Path) -> None:
    probe = {
        "cuquantum": {"enabled": False, "version": "x", "details": {}},
        "stim": {"enabled": True, "version": "x", "details": {}},
        "qujax": {"enabled": False, "version": "x", "details": {}},
        "qhybrid": {"enabled": True, "version": "x", "details": {}},
        "cudaq": {"enabled": False, "version": "x", "details": {}},
    }
    backends = _normalise_sampling_backends("sweep", probe)
    rows: list[BenchmarkRow] = []

    for idx, backend in enumerate(backends):
        metadata = _backend_row_metadata(backend, probe, f"trace-{idx}")
        rows.append(
            BenchmarkRow(
                domain="circuit",
                family="surface",
                decoder="mwpm",
                distance=3,
                physical_error_rate=0.001,
                shots=128,
                metric_name="logical_error_rate",
                metric_value=float(idx),
                backend=backend,
                backend_enabled=metadata["backend_enabled"],
                backend_version=metadata["backend_version"],
                fallback_reason=metadata["fallback_reason"],
                sample_trace_id=metadata["sample_trace_id"],
                backend_chain=metadata["backend_chain"],
                backend_chain_tokens=metadata["backend_chain_tokens"],
                contract_flags=metadata["contract_flags"],
                profiler_flags=metadata["profiler_flags"],
            )
        )

    csv_path = tmp_path / "benchmark.csv"
    json_path = tmp_path / "benchmark.json"

    _write_csv(rows, csv_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([row.__dict__ for row in rows], f, indent=2)

    with csv_path.open(encoding="utf-8", newline="") as f:
        written_csv = list(csv.DictReader(f))
    with json_path.open(encoding="utf-8") as f:
        written_json = json.load(f)

    assert len(written_csv) == len(rows)
    assert len(written_json) == len(rows)

    for row, csv_row, json_row in zip(rows, written_csv, written_json):
        csv_chain_tokens = json.loads(csv_row["backend_chain_tokens"])
        json_chain_tokens = json_row["backend_chain_tokens"]
        if isinstance(json_chain_tokens, str):
            json_chain_tokens = json.loads(json_chain_tokens)
        assert row.backend_chain_tokens is not None
        expected_chain_tokens = row.backend_chain_tokens

        assert isinstance(csv_chain_tokens, list)
        assert isinstance(json_chain_tokens, list)
        assert csv_chain_tokens == expected_chain_tokens
        assert json_chain_tokens == expected_chain_tokens
