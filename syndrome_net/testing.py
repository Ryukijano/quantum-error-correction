"""Testing utilities for syndrome-net.

Provides fixtures, helpers, and property-based testing infrastructure
for comprehensive test coverage.
"""
from __future__ import annotations

from typing import Iterator, Any
from dataclasses import dataclass
import random

import numpy as np
from numpy.typing import NDArray
import stim
import pytest

from syndrome_net import (
    CircuitSpec,
    CircuitBuilder,
    Decoder,
    Correction,
    Syndrome,
    DIContainer,
)
from syndrome_net.container import ContainerConfig, get_container, reset_container


@pytest.fixture
def container() -> DIContainer:
    """Provide a configured DI container for tests."""
    reset_container()
    container = get_container()
    container.register_defaults()
    yield container
    reset_container()


@pytest.fixture
def fresh_container() -> DIContainer:
    """Provide a fresh, empty DI container for tests."""
    return DIContainer()


@pytest.fixture
def sample_circuit_spec() -> CircuitSpec:
    """Provide a sample circuit specification."""
    return CircuitSpec(distance=3, rounds=5, error_probability=0.001)


class CircuitBuilderTests:
    """Mixin class for testing CircuitBuilder implementations.
    
    Provides standard tests that all circuit builders should pass.
    """
    
    builder_class: type[CircuitBuilder]
    
    def test_builds_valid_circuit(self, sample_circuit_spec: CircuitSpec) -> None:
        """Test that builder produces a valid Stim circuit."""
        builder = self.builder_class()
        circuit = builder.build(sample_circuit_spec)
        
        # Should be a valid Stim circuit
        assert isinstance(circuit, stim.Circuit)
        
        # Should have detectors defined
        try:
            circuit.detector_error_model()
        except Exception as e:
            pytest.fail(f"Circuit has non-deterministic detectors: {e}")
    
    def test_supported_distances(self) -> None:
        """Test that supported distances are valid."""
        builder = self.builder_class()
        distances = builder.supported_distances
        
        assert len(distances) > 0
        assert all(isinstance(d, int) for d in distances)
        assert all(d >= 3 for d in distances)
        assert all(d % 2 == 1 for d in distances)  # Odd distances only
    
    def test_invalid_distance_raises(self) -> None:
        """Test that invalid distance raises appropriate error."""
        builder = self.builder_class()
        invalid_dist = 1000  # Assume this is not supported
        
        if invalid_dist not in builder.supported_distances:
            spec = CircuitSpec(distance=invalid_dist, rounds=5, error_probability=0.001)
            
            with pytest.raises((ValueError, Exception)):
                builder.build(spec)
    
    def test_name_is_string(self) -> None:
        """Test that builder name is a string."""
        builder = self.builder_class()
        assert isinstance(builder.name, str)
        assert len(builder.name) > 0


class DecoderTests:
    """Mixin class for testing Decoder implementations."""
    
    decoder_class: type[Decoder]
    
    def test_decodes_empty_syndrome(self) -> None:
        """Test decoder handles empty syndrome."""
        decoder = self.decoder_class()
        syndrome = Syndrome(
            x_syndrome=np.zeros(4, dtype=bool),
            z_syndrome=np.zeros(4, dtype=bool),
            time=0
        )
        
        correction = decoder.decode(syndrome)
        
        assert isinstance(correction, Correction)
        assert correction.x_flips.dtype == bool
        assert correction.z_flips.dtype == bool
    
    def test_reset_clears_state(self) -> None:
        """Test that reset clears decoder state."""
        decoder = self.decoder_class()
        
        # Add some state
        syndrome = Syndrome(
            x_syndrome=np.array([True, False, True, False]),
            z_syndrome=np.array([False, True, False, True]),
            time=0
        )
        decoder.decode(syndrome)
        
        # Reset should clear state
        decoder.reset()
        
        # New decoding should be fresh
        correction = decoder.decode(syndrome)
        assert isinstance(correction, Correction)


@dataclass
class TestDataGenerator:
    """Generate test data for property-based testing."""
    
    seed: int | None = None
    
    def __post_init__(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
    
    def random_syndrome(
        self,
        n_x_checks: int,
        n_z_checks: int,
        p: float = 0.1
    ) -> Syndrome:
        """Generate random syndrome."""
        return Syndrome(
            x_syndrome=np.random.random(n_x_checks) < p,
            z_syndrome=np.random.random(n_z_checks) < p,
            time=0
        )
    
    def random_error_pattern(
        self,
        n_qubits: int,
        p: float = 0.1
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Generate random X and Z error patterns."""
        x_errors = np.random.random(n_qubits) < p
        z_errors = np.random.random(n_qubits) < p
        return x_errors, z_errors
    
    def valid_circuit_specs(
        self,
        builder: CircuitBuilder,
        rounds_range: tuple[int, int] = (3, 20),
        p_range: tuple[float, float] = (0.0001, 0.1)
    ) -> Iterator[CircuitSpec]:
        """Generate valid circuit specs for a builder."""
        for distance in builder.supported_distances[:3]:  # Limit to first 3
            for rounds in range(rounds_range[0], rounds_range[1], 3):
                for p in np.linspace(p_range[0], p_range[1], 5):
                    yield CircuitSpec(
                        distance=distance,
                        rounds=rounds,
                        error_probability=p
                    )


def assert_deterministic_circuit(circuit: stim.Circuit) -> None:
    """Assert that a circuit has deterministic detectors and observables.
    
    Raises:
        AssertionError: If circuit is non-deterministic
    """
    try:
        circuit.detector_error_model()
    except ValueError as e:
        if "non-deterministic" in str(e).lower():
            raise AssertionError(f"Circuit has non-deterministic elements: {e}")
        raise


def assert_valid_correction(
    correction: Correction,
    n_qubits: int
) -> None:
    """Assert that a correction is valid.
    
    Args:
        correction: The correction to validate
        n_qubits: Expected number of qubits
    """
    assert len(correction.x_flips) == n_qubits
    assert len(correction.z_flips) == n_qubits
    assert correction.x_flips.dtype == bool
    assert correction.z_flips.dtype == bool
    assert 0 <= correction.confidence <= 1


def benchmark_decoder(
    decoder: Decoder,
    n_syndromes: int = 1000,
    n_x_checks: int = 16,
    n_z_checks: int = 16
) -> dict[str, float]:
    """Benchmark decoder performance.
    
    Returns:
        Dictionary with timing and accuracy metrics
    """
    import time
    
    generator = TestDataGenerator()
    
    start = time.perf_counter()
    for _ in range(n_syndromes):
        syndrome = generator.random_syndrome(n_x_checks, n_z_checks)
        decoder.decode(syndrome)
    elapsed = time.perf_counter() - start
    
    return {
        "total_time": elapsed,
        "per_syndrome": elapsed / n_syndromes,
        "syndromes_per_second": n_syndromes / elapsed
    }


# Property-based testing with hypothesis (if available)
try:
    from hypothesis import given, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


if HYPOTHESIS_AVAILABLE:
    @given(
        distance=st.sampled_from([3, 5, 7]),
        rounds=st.integers(min_value=3, max_value=20),
        p=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)
    )
    def test_surface_code_properties(
        distance: int,
        rounds: int,
        p: float
    ) -> None:
        """Property-based test for surface code."""
        from syndrome_net.codes import SurfaceCodeBuilder
        
        builder = SurfaceCodeBuilder()
        if distance not in builder.supported_distances:
            return
        
        spec = CircuitSpec(distance=distance, rounds=rounds, error_probability=p)
        circuit = builder.build(spec)
        
        # Should be deterministic
        assert_deterministic_circuit(circuit)
        
        # Should have correct number of qubits
        n_qubits = circuit.num_qubits
        expected_min = distance ** 2
        assert n_qubits >= expected_min
