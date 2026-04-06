"""Regression tests: all circuit builders must produce deterministic Stim circuits.

A deterministic circuit is one where detector_error_model() succeeds without
raising ValueError about non-deterministic detectors or observables.
"""
from __future__ import annotations

import pytest
import stim

from surface_code_in_stem.dynamic import (
    hexagonal_surface_code,
    iswap_surface_code,
    walking_surface_code,
    xyz2_hexagonal_code,
)
from surface_code_in_stem.surface_code import surface_code_circuit_string


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_deterministic(circuit: stim.Circuit, name: str) -> None:
    """Assert circuit has deterministic detectors and observables."""
    try:
        circuit.detector_error_model()
    except ValueError as exc:
        pytest.fail(f"{name} circuit is non-deterministic: {exc}")


def _build_surface(distance: int, rounds: int, p: float) -> stim.Circuit:
    return stim.Circuit(surface_code_circuit_string(distance=distance, rounds=rounds, p=p))


def _build_hexagonal(distance: int, rounds: int, p: float) -> stim.Circuit:
    return stim.Circuit(str(hexagonal_surface_code(distance=distance, rounds=rounds, p=p)))


def _build_walking(distance: int, rounds: int, p: float) -> stim.Circuit:
    return stim.Circuit(str(walking_surface_code(distance=distance, rounds=rounds, p=p)))


def _build_iswap(distance: int, rounds: int, p: float) -> stim.Circuit:
    return stim.Circuit(str(iswap_surface_code(distance=distance, rounds=rounds, p=p)))


def _build_xyz2(distance: int, rounds: int, p: float) -> stim.Circuit:
    return xyz2_hexagonal_code(distance=distance, rounds=rounds, p=p)


# Colour code builders (optional dependencies)
def _build_color_code(distance: int, rounds: int, p: float) -> stim.Circuit:
    """Build colour code circuit using color-code-stim."""
    try:
        from color_code_stim import ColorCode, NoiseModel as CCNoiseModel
        from syndrome_net import CircuitSpec
        from syndrome_net.codes import ColorCodeStimBuilder
        
        spec = CircuitSpec(distance=distance, rounds=rounds, error_probability=p, circuit_type="tri")
        builder = ColorCodeStimBuilder()
        return builder.build(spec)
    except ImportError as exc:
        pytest.skip(f"color-code-stim not installed: {exc}")


def _build_loom_color_code(distance: int, rounds: int, p: float) -> stim.Circuit:
    """Build colour code circuit using el-loom."""
    try:
        from syndrome_net import CircuitSpec
        from syndrome_net.codes import LoomColorCodeBuilder
        
        spec = CircuitSpec(distance=distance, rounds=rounds, error_probability=p)
        builder = LoomColorCodeBuilder()
        return builder.build(spec)
    except ImportError as exc:
        pytest.skip(f"el-loom not installed: {exc}")


# ---------------------------------------------------------------------------
# Parametrized determinism tests
# ---------------------------------------------------------------------------

BUILDERS = {
    "surface": _build_surface,
    "hexagonal": _build_hexagonal,
    "walking": _build_walking,
    "iswap": _build_iswap,
    "xyz2": _build_xyz2,
}

COLOUR_CODE_BUILDERS = {
    "color_code": _build_color_code,
    "loom_color_code": _build_loom_color_code,
}

DISTANCES = [3, 5]
ROUNDS = [3, 7]
P_VALUES = [0.0, 0.001, 0.01]


@pytest.mark.parametrize("name,builder", BUILDERS.items())
@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("rounds", [3])
@pytest.mark.parametrize("p", [0.001])
def test_deterministic(name: str, builder, distance: int, rounds: int, p: float) -> None:
    """Every code family must build a deterministic circuit."""
    circuit = builder(distance=distance, rounds=rounds, p=p)
    _check_deterministic(circuit, f"{name}(d={distance},r={rounds},p={p})")


@pytest.mark.parametrize("name,builder", BUILDERS.items())
def test_deterministic_noiseless(name: str, builder) -> None:
    """Noiseless circuits (p=0) must also be deterministic."""
    circuit = builder(distance=3, rounds=3, p=0.0)
    _check_deterministic(circuit, f"{name}(p=0)")


@pytest.mark.parametrize("name,builder", BUILDERS.items())
def test_has_detectors(name: str, builder) -> None:
    """Circuits must define at least one detector."""
    circuit = builder(distance=3, rounds=3, p=0.001)
    assert circuit.num_detectors > 0, f"{name}: no detectors defined"


@pytest.mark.parametrize("name,builder", BUILDERS.items())
def test_has_observable(name: str, builder) -> None:
    """Circuits must define exactly one logical observable."""
    circuit = builder(distance=3, rounds=3, p=0.001)
    assert circuit.num_observables == 1, (
        f"{name}: expected 1 observable, got {circuit.num_observables}"
    )


@pytest.mark.parametrize("name,builder", BUILDERS.items())
def test_sample_without_error(name: str, builder) -> None:
    """Monte Carlo sampling must succeed without exceptions."""
    circuit = builder(distance=3, rounds=3, p=0.01)
    sampler = circuit.compile_detector_sampler(seed=42)
    det, obs = sampler.sample(100, separate_observables=True)
    assert det.shape[0] == 100
    assert obs.shape[0] == 100


# ---------------------------------------------------------------------------
# Colour code determinism tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,builder", COLOUR_CODE_BUILDERS.items())
@pytest.mark.parametrize("distance", [3, 5])  # Odd distances for triangular
@pytest.mark.parametrize("p", [0.001, 0.01])
def test_colour_code_deterministic(name: str, builder, distance: int, p: float) -> None:
    """Colour code circuits must be deterministic."""
    circuit = builder(distance=distance, rounds=distance, p=p)
    _check_deterministic(circuit, f"{name}(d={distance},p={p})")


@pytest.mark.parametrize("name,builder", COLOUR_CODE_BUILDERS.items())
def test_colour_code_has_detectors(name: str, builder) -> None:
    """Colour code circuits must define detectors."""
    circuit = builder(distance=3, rounds=3, p=0.001)
    assert circuit.num_detectors > 0, f"{name}: no detectors defined"


@pytest.mark.parametrize("name,builder", COLOUR_CODE_BUILDERS.items())
def test_colour_code_has_observable(name: str, builder) -> None:
    """Colour code circuits must define one logical observable."""
    circuit = builder(distance=3, rounds=3, p=0.001)
    assert circuit.num_observables == 1, (
        f"{name}: expected 1 observable, got {circuit.num_observables}"
    )


@pytest.mark.parametrize("name,builder", COLOUR_CODE_BUILDERS.items())
def test_colour_code_sample_without_error(name: str, builder) -> None:
    """Colour code Monte Carlo sampling must succeed."""
    circuit = builder(distance=3, rounds=3, p=0.01)
    sampler = circuit.compile_detector_sampler(seed=42)
    det, obs = sampler.sample(100, separate_observables=True)
    assert det.shape[0] == 100
    assert obs.shape[0] == 100


# ---------------------------------------------------------------------------
# Larger distance smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,builder", BUILDERS.items())
def test_distance_5_deterministic(name: str, builder) -> None:
    """Distance-5 circuits must also be deterministic."""
    circuit = builder(distance=5, rounds=5, p=0.001)
    _check_deterministic(circuit, f"{name}(d=5)")
