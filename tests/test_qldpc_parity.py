"""Tests for qLDPC parity-check matrix code generation."""

import pytest
import numpy as np

stim = pytest.importorskip("stim")
from codes.qldpc.parity_builder import (
    qldpc_from_parity_matrices,
    toric_code_parity,
    surface_code_parity,
    hamming_code_parity,
    hypergraph_product
)


def test_toric_code_parity():
    # 3x3 toric code
    hx, hz = toric_code_parity(3)
    
    # Check dimensions
    num_qubits = 2 * 3 * 3  # 18
    num_stabilizers = 3 * 3  # 9
    
    assert hx.shape == (num_stabilizers, num_qubits)
    assert hz.shape == (num_stabilizers, num_qubits)
    
    # Check CSS condition: H_x @ H_z^T = 0 mod 2
    assert np.all((hx @ hz.T) % 2 == 0)


def test_surface_code_parity():
    hx, hz = surface_code_parity(3)
    
    assert hx.shape[1] == hz.shape[1]  # Same number of qubits
    
    # Check CSS condition
    assert np.all((hx @ hz.T) % 2 == 0)


def test_hamming_code_parity():
    h = hamming_code_parity(3)
    
    # Dimensions for r=3: 3 x 7
    assert h.shape == (3, 7)
    
    # Check all columns are unique and non-zero
    cols = set(tuple(h[:, i]) for i in range(7))
    assert len(cols) == 7
    assert (0, 0, 0) not in cols


def test_hypergraph_product():
    h1 = hamming_code_parity(3)  # 3x7
    h2 = hamming_code_parity(3)  # 3x7
    
    hx, hz = hypergraph_product(h1, h2)
    
    # Number of qubits = 7*7 + 3*3 = 58
    assert hx.shape[1] == 58
    assert hz.shape[1] == 58
    
    # Number of X checks for 3x7 and 3x7 = r1*n2 + r2*n1 = 3*7 + 3*7 = 42?
    # Actually wait: The hypergraph product we built uses np.hstack with np.kron.
    # The output of np.kron(h1, np.eye(n2)) has shape (3*7, 7*7) = (21, 49)
    # The output of np.kron(np.eye(r1), h2.T) has shape (3*7, 3*3) = (21, 9)
    # The hstack gives shape (21, 58).
    # Ah, the number of checks is just r1*n2 = 21 for X and n1*r2 = 21 for Z in this specific definition!
    
    # Number of X checks = r1*n2 = 3*7 = 21
    assert hx.shape[0] == 21
    
    # Number of Z checks = n1*r2 = 7*3 = 21
    assert hz.shape[0] == 21
    
    # Check CSS condition
    assert np.all((hx @ hz.T) % 2 == 0)


def test_qldpc_from_parity_matrices():
    hx, hz = toric_code_parity(2)
    
    circuit = qldpc_from_parity_matrices(
        hx=hx,
        hz=hz,
        rounds=2,
        p=0.001
    )
    
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_detectors > 0
    
    # Check it runs without errors
    sampler = circuit.compile_detector_sampler()
    sample = sampler.sample(1)
    assert sample.shape[0] == 1
