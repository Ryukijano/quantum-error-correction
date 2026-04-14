from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .noise import apply_kraus_1q_density_matrix


def kraus_1q_from_qiskit(error: object) -> NDArray[np.complex128]:
    """
    Extract 1-qubit Kraus operators from a Qiskit object.

    Works for many Qiskit quantum channels and Aer noise errors via `qiskit.quantum_info.Kraus`.
    Returns a complex ndarray of shape (k, 2, 2).
    """
    try:
        from qiskit.quantum_info import Kraus  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Qiskit is not installed. Install with `pip install qhybrid-kernels[qiskit]`."
        ) from e

    k = Kraus(error)
    data = np.asarray(k.data, dtype=np.complex128)

    # Qiskit may return shape (k, 2, 2) or (2, 2) for a single Kraus op
    if data.shape == (2, 2):
        data = data[None, :, :]

    if data.ndim != 3 or data.shape[1:] != (2, 2):
        raise ValueError(f"Expected Kraus ops of shape (k,2,2); got {data.shape}")

    return data


def apply_qiskit_kraus_1q_to_density_matrix(
    rho: ArrayLike,
    n_qubits: int,
    target_qubit: int,
    error: object,
) -> NDArray[np.complex128]:
    """
    Convenience wrapper:
      - convert Qiskit error/channel to Kraus
      - call the Rust kernel
    """
    kraus_ops = kraus_1q_from_qiskit(error)
    return apply_kraus_1q_density_matrix(rho, n_qubits, target_qubit, kraus_ops)


