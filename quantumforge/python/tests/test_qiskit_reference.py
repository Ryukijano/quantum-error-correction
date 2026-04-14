import numpy as np
import pytest


pytest.importorskip("qiskit")


def _amplitude_damping_kraus(gamma: float) -> np.ndarray:
    g = float(gamma)
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - g)]], dtype=np.complex128)
    k1 = np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], dtype=np.complex128)
    return np.stack([k0, k1], axis=0)


def test_kraus_amplitude_damping_matches_qiskit_quantum_info():
    # This test uses Qiskit quantum_info as a reference implementation.
    from qiskit.quantum_info import DensityMatrix, Kraus

    from qhybrid_kernels import apply_kraus_1q_density_matrix

    gamma = 0.3
    kraus = _amplitude_damping_kraus(gamma)

    # Start from |1><1|
    rho = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    out = apply_kraus_1q_density_matrix(rho, n_qubits=1, target_qubit=0, kraus_ops=kraus)

    # Qiskit reference: evolve density matrix under Kraus channel
    channel = Kraus(kraus)
    ref = DensityMatrix(rho).evolve(channel).data

    assert np.allclose(out, ref, atol=1e-12, rtol=1e-12)


