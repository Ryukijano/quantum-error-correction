"""Shared pytest fixtures for Syndrome-Net tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

try:
    import stim
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    stim = None

from surface_code_in_stem.surface_code import surface_code_circuit_string


@pytest.fixture(scope="session")
def surface_code_circuit() -> stim.Circuit:
    if stim is None:
        pytest.skip("stim is required for circuit fixtures")
    return stim.Circuit(surface_code_circuit_string(distance=3, rounds=2, p=0.001))


@pytest.fixture(scope="session")
def sample_surface_data(surface_code_circuit: stim.Circuit):
    sampler = surface_code_circuit.compile_detector_sampler(seed=11)
    detector_samples, observable_samples = sampler.sample(8, separate_observables=True)
    return np.asarray(detector_samples, dtype=np.bool_), np.asarray(observable_samples, dtype=np.bool_)


@pytest.fixture(scope="session")
def qldpc_parity_matrices():
    from codes.qldpc.parity_builder import toric_code_parity
    return toric_code_parity(3)


@pytest.fixture()
def benchmark_config():
    from codes import CircuitGenerationConfig
    return CircuitGenerationConfig(
        distance=3,
        rounds=2,
        physical_error_rate=0.001,
        extra_params={"variant": "static"},
    )


@pytest.fixture()
def tmp_artifact_dir(tmp_path: Path) -> Path:
    path = tmp_path / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path
