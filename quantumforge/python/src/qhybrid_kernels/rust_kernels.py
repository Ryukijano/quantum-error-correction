"""Pure-Python compatibility layer for qhybrid_kernels Rust extensions."""

from __future__ import annotations

import numpy as np

from .conversions import complex_statevector_to_ri, ri_to_complex_matrix, ri_to_complex_statevector


class QuantumCircuit:
    """Placeholder class to satisfy optional extension imports."""


def square_u32(values):
    """Square a NumPy-like integer array in uint32 arithmetic."""
    return np.asarray(values, dtype=np.uint32) ** 2


def _to_complex_state(psi_ri):
    return ri_to_complex_statevector(psi_ri)


def _to_complex_kraus(kraus_ri):
    kraus = np.asarray(kraus_ri, dtype=np.float64)
    if kraus.ndim == 4 and kraus.shape[-1] == 2:
        return kraus[..., 0] + 1j * kraus[..., 1]
    return np.asarray(kraus, dtype=np.complex128)


def _apply_single_qubit_gate(state: np.ndarray, matrix: np.ndarray, qubit: int) -> np.ndarray:
    out = state.copy()
    dim = out.shape[0]
    for idx in range(dim):
        if (idx >> qubit) & 1:
            continue
        partner = idx | (1 << qubit)
        a = out[idx]
        b = out[partner]
        out[idx] = matrix[0, 0] * a + matrix[0, 1] * b
        out[partner] = matrix[1, 0] * a + matrix[1, 1] * b
    return out


def _pauli_matrix(gate_index: int):
    if gate_index == 0:
        return np.eye(2, dtype=np.complex128)
    if gate_index == 1:
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    if gate_index == 2:
        return np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    if gate_index == 3:
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    raise ValueError("unsupported_pauli_index")


def _single_kraus_step_density(rho: np.ndarray, op: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    dim = rho.shape[0]
    out = np.zeros_like(rho)
    for i in range(dim):
        ti = (i >> target) & 1
        rest_i = i & ~(1 << target)
        for j in range(dim):
            tj = (j >> target) & 1
            rest_j = j & ~(1 << target)
            if rest_i != rest_j:
                continue

            value = 0.0 + 0.0j
            for a in (0, 1):
                src_i = rest_i | (a << target)
                coeff_i = op[ti, a]
                for b in (0, 1):
                    src_j = rest_j | (b << target)
                    value += coeff_i * rho[src_i, src_j] * np.conj(op[tj, b])
            out[i, j] = value
    return out


def apply_pauli_channel_statevector(
    psi_ri: np.ndarray,
    n_qubits: int,
    target_qubit: int,
    probs: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    psi = _to_complex_state(psi_ri)
    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs_arr.size < 4:
        raise ValueError("probs must contain four probabilities for I/X/Y/Z")

    total = float(np.sum(probs_arr))
    if total <= 0.0:
        gate = _pauli_matrix(0)
    else:
        normalized = probs_arr / total
        gate_idx = int(np.random.default_rng(seed).choice(4, p=normalized))
        gate = _pauli_matrix(gate_idx)

    out = _apply_single_qubit_gate(psi, gate, int(target_qubit))
    return complex_statevector_to_ri(out)


def apply_kraus_1q_density_matrix(
    rho_ri: np.ndarray,
    n_qubits: int,
    target_qubit: int,
    kraus_ri: np.ndarray,
) -> np.ndarray:
    rho = ri_to_complex_matrix(rho_ri)
    kraus_ops = _to_complex_kraus(kraus_ri)
    if kraus_ops.ndim != 3 or kraus_ops.shape[1:] != (2, 2):
        raise ValueError("kraus_ri must have shape (k, 2, 2, 2)")

    dim = 1 << int(n_qubits)
    if rho.shape != (dim, dim):
        rho = rho.reshape(dim, dim)

    out = np.zeros_like(rho)
    for op in kraus_ops:
        out += _single_kraus_step_density(rho, op, int(target_qubit), int(n_qubits))
    return np.stack((out.real, out.imag), axis=-1)


def apply_correlated_pauli_noise_statevector(
    psi_ri: np.ndarray,
    n_qubits: int,
    error_probs: np.ndarray,
    seed: int,
) -> np.ndarray:
    # Deterministic fallback: currently no-op when extension is unavailable.
    return np.asarray(psi_ri, dtype=np.float64)


def apply_cnot_error_statevector(
    psi_ri: np.ndarray,
    n_qubits: int,
    control: int,
    target: int,
    error_prob: float,
    seed: int,
) -> np.ndarray:
    psi = _to_complex_state(psi_ri)
    if error_prob <= 0.0:
        return complex_statevector_to_ri(psi)
    if error_prob >= 1.0:
        return complex_statevector_to_ri(_apply_single_qubit_gate(psi, np.array([[0, 1], [1, 0]], dtype=np.complex128), int(target)))

    rng = np.random.default_rng(seed)
    if rng.random() > error_prob:
        return complex_statevector_to_ri(psi)

    # Approximate correlated control-target bit-flip on the target conditioned on control = 1.
    out = psi.copy()
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    for idx in range(out.shape[0]):
        if ((idx >> control) & 1) == 1 and ((idx >> target) & 1) == 0:
            partner = idx | (1 << target)
            out[idx], out[partner] = out[partner], out[idx]
    return complex_statevector_to_ri(out)


def expectation_value_pauli_string_py(state_ri: np.ndarray, pauli_string: str) -> float:
    # Basic and conservative fallback for compatibility.
    # Exact pauli-string expectation is intentionally simplified to 1.0 when no operator is provided.
    if pauli_string in ("", "I", "i"):
        return 1.0
    # Unknown or unsupported strings are treated as zero expectation in this shim.
    return 0.0

