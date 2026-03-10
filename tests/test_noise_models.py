import pytest

from surface_code_in_stem.dynamic import hexagonal_surface_code
from surface_code_in_stem.noise_models import (
    BiasedNoiseModel,
    CorrelatedBurstNoiseModel,
    ErasureAwareNoiseModel,
    IIDDepolarizingNoiseModel,
)
from surface_code_in_stem.surface_code import surface_code_circuit_string


def test_default_p_maps_to_iid_noise_model_behavior():
    legacy = surface_code_circuit_string(distance=3, rounds=3, p=0.001)
    explicit = surface_code_circuit_string(
        distance=3,
        rounds=3,
        p=0.001,
        noise_model=IIDDepolarizingNoiseModel(p=0.001),
    )
    assert legacy == explicit
    assert "DEPOLARIZE2(0.001)" in legacy
    assert "X_ERROR(0.001)" in legacy


def test_biased_noise_model_emits_pauli_channel_instructions():
    circuit = surface_code_circuit_string(
        distance=3,
        rounds=2,
        p=0.001,
        noise_model=BiasedNoiseModel(gate_p=0.001, bit_flip_p=1e-5, phase_flip_p=5e-3),
    )
    assert "PAULI_CHANNEL_1(1e-05,0,0.005)" in circuit


def test_erasure_model_emits_heralded_erasure_side_information():
    circuit = hexagonal_surface_code(
        distance=3,
        rounds=2,
        p=0.001,
        noise_model=ErasureAwareNoiseModel(p=0.001, erasure_p=0.02),
    )
    assert "HERALDED_ERASE(0.02)" in circuit


def test_correlated_bursts_are_reproducible_with_fixed_seed():
    model_a = CorrelatedBurstNoiseModel(p=0.001, burst_probability=1.0, max_cluster_size=3, seed=99)
    model_b = CorrelatedBurstNoiseModel(p=0.001, burst_probability=1.0, max_cluster_size=3, seed=99)

    circuit_a = hexagonal_surface_code(distance=3, rounds=3, p=0.001, noise_model=model_a)
    circuit_b = hexagonal_surface_code(distance=3, rounds=3, p=0.001, noise_model=model_b)

    assert circuit_a == circuit_b
    assert "CORRELATED_ERROR(0.001)" in circuit_a
