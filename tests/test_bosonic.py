"""Tests for the bosonic code variants."""

import pytest

stim = pytest.importorskip("stim")
from codes.bosonic.cat_code import cat_surface_code, cat_biased_circuit
from codes.bosonic.squeezed_state import squeezed_surface_code, squeezed_circuit_string


def test_cat_surface_code():
    distance = 3
    rounds = 2
    p = 0.001
    
    circuit = cat_surface_code(distance, rounds, p, alpha=2.0)
    
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_detectors > 0
    assert circuit.num_observables > 0
    
    # Check that it runs without errors
    sampler = circuit.compile_detector_sampler(seed=42)
    sample = sampler.sample(1)
    assert sample.shape[0] == 1


def test_cat_biased_circuit_string():
    distance = 3
    rounds = 2
    p = 0.001
    
    circuit_str = cat_biased_circuit(distance, rounds, p, alpha=2.0)
    
    assert isinstance(circuit_str, str)
    assert "# Cat Code Surface Code" in circuit_str
    
    # Check it compiles back
    circuit = stim.Circuit(circuit_str)
    assert circuit.num_detectors > 0


def test_squeezed_surface_code():
    distance = 3
    rounds = 2
    p = 0.001
    
    circuit = squeezed_surface_code(distance, rounds, p, squeezing_db=10.0, squeezed_quadrature="p")
    
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_detectors > 0
    assert circuit.num_observables > 0


def test_squeezed_circuit_string():
    distance = 3
    rounds = 2
    p = 0.001
    
    circuit_str = squeezed_circuit_string(distance, rounds, p, squeezing_db=10.0, squeezed_quadrature="q")
    
    assert isinstance(circuit_str, str)
    assert "# Squeezed State Surface Code" in circuit_str
    assert "q-squeezed" in circuit_str
    
    circuit = stim.Circuit(circuit_str)
    assert circuit.num_detectors > 0
