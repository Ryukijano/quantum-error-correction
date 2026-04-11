from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conversions import (
    complex_matrix_to_ri,
    complex_statevector_to_ri,
    kraus_1q_to_ri,
    ri_to_complex_matrix,
    ri_to_complex_statevector,
)

try:
    from . import rust_kernels as _rk  # built by maturin from ../rust_kernels
except Exception as e:  # pragma: no cover
    _rk = None
    _import_error = e


def _require_rust_kernels():
    if _rk is None:  # pragma: no cover
        raise ImportError(
            "rust_kernels extension is not available. "
            "Did you run `maturin develop` in quantumforge/python/?"
        ) from _import_error


def apply_pauli_channel_statevector(
    psi: ArrayLike,
    n_qubits: int,
    target_qubit: int,
    probs: ArrayLike,
    seed: int = 0,
) -> NDArray[np.complex128]:
    """
    Apply a 1-qubit Pauli channel to a statevector by sampling one of {I,X,Y,Z}.
    """
    _require_rust_kernels()
    psi_ri = complex_statevector_to_ri(psi)
    probs_f = np.asarray(probs, dtype=np.float64)
    out_ri = _rk.apply_pauli_channel_statevector(
        psi_ri, int(n_qubits), int(target_qubit), probs_f, int(seed)
    )
    return ri_to_complex_statevector(out_ri)


def apply_kraus_1q_density_matrix(
    rho: ArrayLike,
    n_qubits: int,
    target_qubit: int,
    kraus_ops: ArrayLike,
) -> NDArray[np.complex128]:
    """
    Apply a 1-qubit Kraus channel to a density matrix: rho' = Σ K rho K†.
    """
    _require_rust_kernels()
    rho_ri = complex_matrix_to_ri(rho)
    kraus_ri = kraus_1q_to_ri(kraus_ops)
    out_ri = _rk.apply_kraus_1q_density_matrix(
        rho_ri, int(n_qubits), int(target_qubit), kraus_ri
    )
    return ri_to_complex_matrix(out_ri)


def apply_correlated_pauli_noise_statevector(
    psi: ArrayLike,
    n_qubits: int,
    error_probs: ArrayLike,
    seed: int,
) -> NDArray[np.complex128]:
    """Apply correlated multi-qubit Pauli noise to a statevector.

    Args:
        psi: Complex statevector (shape will be inferred)
        n_qubits: Number of qubits
        error_probs: Correlation matrix (2^n x 2^n) of error probabilities
        seed: Random seed for reproducibility

    Returns:
        Noisy statevector (same shape as input)
    """
    _require_rust_kernels()

    psi_ri = complex_statevector_to_ri(psi)
    error_probs_array = np.asarray(error_probs, dtype=np.float64)

    out_ri = _rk.apply_correlated_pauli_noise_statevector(
        psi_ri, n_qubits, error_probs_array, seed
    )
    return ri_to_complex_statevector(out_ri)


def apply_cnot_error_statevector(
    psi: ArrayLike,
    n_qubits: int,
    control: int,
    target: int,
    error_prob: float,
    seed: int,
) -> NDArray[np.complex128]:
    """Apply CNOT gate error (correlated bit flips) to a statevector.

    Args:
        psi: Complex statevector
        n_qubits: Number of qubits
        control: Control qubit index
        target: Target qubit index
        error_prob: Probability of correlated error
        seed: Random seed

    Returns:
        Statevector with potential CNOT error applied
    """
    _require_rust_kernels()

    psi_ri = complex_statevector_to_ri(psi)

    out_ri = _rk.apply_cnot_error_statevector(
        psi_ri, n_qubits, control, target, error_prob, seed
    )
    return ri_to_complex_statevector(out_ri)


def expectation_value_pauli_string_py(
    state: ArrayLike,
    pauli_string: str,
) -> float:
    """Compute expectation value ⟨ψ|P|ψ⟩ for a Pauli string P.

    Args:
        state: Complex statevector
        pauli_string: String like "XYZI" (one Pauli per qubit)

    Returns:
        Expectation value (float)
    """
    _require_rust_kernels()

    state_ri = complex_statevector_to_ri(state)
    return _rk.expectation_value_pauli_string_py(state_ri, pauli_string)


