import numpy as np


def test_square_u32():
    import qhybrid_kernels.rust_kernels as rk

    x = np.array([1, 2, 3], dtype=np.uint32)
    out = rk.square_u32(x)
    assert out.dtype == np.uint32
    assert np.array_equal(out, np.array([1, 4, 9], dtype=np.uint32))


def test_pauli_x_is_deterministic_when_prob_is_one():
    from qhybrid_kernels import apply_pauli_channel_statevector

    psi = np.array([1 + 0j, 0 + 0j], dtype=np.complex128)
    out = apply_pauli_channel_statevector(
        psi, n_qubits=1, target_qubit=0, probs=[0.0, 1.0, 0.0, 0.0], seed=123
    )
    assert np.allclose(out, np.array([0 + 0j, 1 + 0j], dtype=np.complex128))


def test_kraus_identity_is_noop():
    from qhybrid_kernels import apply_kraus_1q_density_matrix

    rho = np.array([[1 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]], dtype=np.complex128)
    kraus = np.array([[[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]]], dtype=np.complex128)
    out = apply_kraus_1q_density_matrix(rho, n_qubits=1, target_qubit=0, kraus_ops=kraus)
    assert np.allclose(out, rho)


